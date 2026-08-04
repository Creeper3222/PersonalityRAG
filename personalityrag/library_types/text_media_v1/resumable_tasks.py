from __future__ import annotations

import json
import shutil
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...database_types import TEXT_MEDIA_V1_TYPE, DatabaseRef, database_type_registry
from ...database_types import DATABASE_CATEGORY_KNOWLEDGE
from ...io_utils import atomic_write_json, run_blocking
from ...providers import build_provider, build_rerank_provider, provider_kind
from ...task_control import JobExecutionContext, ResolvedJobOperation
from .indexes import TextMediaIndex
from .batch_package import extract_tmkbs
from .package import prepare_tmkb_install
from .service import TextMediaService
from .storage import DATABASE_FILENAME, TextMediaStorage

if TYPE_CHECKING:
    from .manager import TextMediaV1Manager


def _sqlite_backup(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_db = sqlite3.connect(source)
    target_db = sqlite3.connect(target)
    try:
        source_db.backup(target_db)
    finally:
        target_db.close()
        source_db.close()


def _copy_library_snapshot(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)

    def ignore(path: str, names: list[str]) -> set[str]:
        current = Path(path)
        ignored: set[str] = set()
        if current == source:
            ignored.update(
                {
                    DATABASE_FILENAME,
                    f"{DATABASE_FILENAME}-wal",
                    f"{DATABASE_FILENAME}-shm",
                    "task_checkpoints",
                }
            )
        if current.name == "derived":
            ignored.add("indexes")
        return ignored.intersection(names)

    shutil.copytree(source, target, ignore=ignore)
    _sqlite_backup(source / DATABASE_FILENAME, target / DATABASE_FILENAME)


class ResumableTextMediaTasks:
    """Durable text_media_v1 operations with whole-library atomic activation."""

    MUTATION_KINDS = frozenset(
        {
            "text_media_document_ingest",
            "text_media_ingest_batch",
            "text_media_entry_create",
            "text_media_entry_update",
            "text_media_document_delete",
            "text_media_entry_delete",
            "text_media_media_calibration",
            "text_media_media_descriptions_update",
        }
    )
    EMBEDDING_MUTATION_KINDS = frozenset(
        {
            "text_media_document_ingest",
            "text_media_ingest_batch",
            "text_media_entry_create",
            "text_media_entry_update",
            "text_media_media_calibration",
            "text_media_media_descriptions_update",
        }
    )
    IMPORT_KINDS = frozenset({"tmkb_import", "tmkbs_import"})
    INSTALL_OWNER_FILENAME = ".personalityrag-task-owner.json"

    def __init__(self, manager: "TextMediaV1Manager"):
        self.manager = manager

    def workspace(self, job: dict[str, Any]) -> Path:
        return (
            self.manager.data_dir
            / "task_workspaces"
            / TEXT_MEDIA_V1_TYPE
            / str(job["id"])
        )

    async def resolve(self, job: dict[str, Any]) -> ResolvedJobOperation:
        kind = str(job.get("kind") or "")
        if kind == "text_media_index_rebuild":
            run = self._run_index_rebuild
            rollback = self._rollback
            cancel_queued = self._cancel_queued
            finalize_completed = self._finalize_completed
        elif kind in self.MUTATION_KINDS:
            run = self._run_mutation
            rollback = self._rollback
            cancel_queued = self._cancel_queued
            finalize_completed = self._finalize_completed
        elif kind in self.IMPORT_KINDS:
            run = self._run_import
            rollback = self._rollback_import
            cancel_queued = self._cancel_queued_import
            finalize_completed = self._finalize_import
        else:
            raise RuntimeError(f"unsupported text_media_v1 resumable task: {kind}")
        return ResolvedJobOperation(
            run=run,
            rollback=rollback,
            cancel_queued=cancel_queued,
            finalize_completed=finalize_completed,
        )

    async def _state(
        self, context: JobExecutionContext
    ) -> tuple[dict[str, Any], Path]:
        job = await context.job()
        database_id = str(job.get("database_id") or "")
        if not database_id:
            raise ValueError("text media task is missing database identity")
        workspace = self.workspace(job)
        state_path = workspace / "state.json"
        if state_path.exists():
            return json.loads(state_path.read_text(encoding="utf-8")), workspace
        workspace.mkdir(parents=True, exist_ok=True)
        live = database_type_registry.data_dir(
            self.manager.data_dir,
            DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id),
        )
        state = {
            "version": 1,
            "database_id": database_id,
            "phase": "created",
            "live": str(live),
            "stage": str(workspace / "stage"),
            "previous": str(workspace / "previous"),
            "created_at": time.time(),
        }
        await run_blocking(atomic_write_json, state_path, state)
        return state, workspace

    @staticmethod
    async def _save_state(
        workspace: Path,
        state: dict[str, Any],
        *,
        phase: str,
        **values: Any,
    ) -> None:
        state.update(values)
        state["phase"] = phase
        state["updated_at"] = time.time()
        await run_blocking(atomic_write_json, workspace / "state.json", state)

    async def _long_task_embedding_provider(
        self,
        *,
        context: JobExecutionContext,
        state: dict[str, Any],
        workspace: Path,
        provider_id: str,
        expected_revision: int = 0,
        expected_fingerprint: str = "",
    ) -> tuple[Any, dict[str, Any]]:
        pinned = state.get("embedding_context")
        if isinstance(pinned, dict) and pinned.get("provider_fingerprint"):
            revision = await self.manager.control.get_provider(
                str(pinned["provider_id"]),
                int(pinned["provider_revision"]),
            )
            if (
                revision is None
                or not revision.config.enabled
                or provider_kind(revision.config.type) != "embedding"
                or revision.config_sha256
                != str(pinned["provider_fingerprint"])
            ):
                raise ValueError(
                    "the task-pinned Embedding Provider capability is no longer available"
                )
            return revision, pinned

        await context.progress(
            0.005,
            "正在确认 Embedding Provider 上下文长度",
        )
        revision, capability = (
            await self.manager.prepare_embedding_context_for_long_task(
                provider_id,
                expected_revision=expected_revision,
                expected_fingerprint=expected_fingerprint,
            )
        )
        await self._save_state(
            workspace,
            state,
            phase=str(state.get("phase") or "created"),
            embedding_context=capability,
        )
        return revision, capability

    async def _current_embedding_binding(
        self, root: Path
    ) -> dict[str, Any]:
        storage = TextMediaStorage(root)
        await storage.initialize()
        try:
            return await storage.metadata()
        finally:
            await storage.close()

    async def _stage_service(
        self,
        stage: Path,
        *,
        target_provider: Any,
        strict_rerank: bool = True,
    ) -> TextMediaService:
        storage = TextMediaStorage(stage)
        await storage.initialize()
        meta = await storage.metadata()
        reranker = None
        rerank_info: dict[str, Any] = {}
        rerank_id = str(meta.get("rerank_provider_id") or "")
        if rerank_id:
            rerank_revision = await self.manager.control.get_provider(
                rerank_id,
                int(meta.get("rerank_provider_revision") or 0),
            )
            if (
                rerank_revision is None
                or not rerank_revision.config.enabled
                or provider_kind(rerank_revision.config.type) != "rerank"
                or rerank_revision.config_sha256
                != str(meta.get("rerank_provider_fingerprint") or "")
            ):
                if strict_rerank:
                    await storage.close()
                    await target_provider.close()
                    raise ValueError(
                        "bound Rerank Provider is unavailable; task rolled back"
                    )
            else:
                reranker = build_rerank_provider(rerank_revision.config)
                rerank_info = {
                    "id": rerank_revision.provider_id,
                    "revision": rerank_revision.revision,
                    "fingerprint": rerank_revision.config_sha256,
                }
        indexes = TextMediaIndex(stage)
        await indexes.load()
        return TextMediaService(
            stage,
            storage,
            indexes,
            target_provider,
            reranker,
            rerank_info,
        )

    async def _stage_service_for_current_binding(
        self, stage: Path, *, strict_rerank: bool
    ) -> TextMediaService:
        probe = TextMediaStorage(stage)
        await probe.initialize()
        try:
            meta = await probe.metadata()
        finally:
            await probe.close()
        revision = await self.manager.control.get_provider(
            str(meta.get("provider_id") or ""),
            int(meta.get("provider_revision") or 0),
        )
        if (
            revision is None
            or not revision.config.enabled
            or provider_kind(revision.config.type) != "embedding"
            or revision.config_sha256 != str(meta.get("provider_fingerprint") or "")
        ):
            raise ValueError(
                "bound Embedding Provider is unavailable for the text media task"
            )
        return await self._stage_service(
            stage,
            target_provider=build_provider(revision.config),
            strict_rerank=strict_rerank,
        )

    @staticmethod
    def _operation_input_root(job: dict[str, Any]) -> Path | None:
        value = str((job.get("operation") or {}).get("input_root") or "").strip()
        return Path(value) if value else None

    def _allowed_task_path(self, path: Path) -> bool:
        candidate = path.resolve()
        allowed_roots = (
            (
                self.manager.data_dir
                / "task_workspaces"
                / TEXT_MEDIA_V1_TYPE
            ).resolve(),
            (
                self.manager.data_dir
                / "task_inputs"
                / TEXT_MEDIA_V1_TYPE
            ).resolve(),
        )
        return any(
            candidate != root and root in candidate.parents
            for root in allowed_roots
        )

    async def _cleanup_path(self, path: Path | None) -> None:
        if path is None or not path.exists():
            return
        if not self._allowed_task_path(path):
            raise RuntimeError(f"refusing to clean non-task path: {path}")
        await run_blocking(shutil.rmtree, path, True)

    async def _cleanup_import_package(self, operation: dict[str, Any]) -> None:
        raw = str(operation.get("package_path") or "").strip()
        if not raw:
            return
        path = Path(raw).resolve()
        allowed = (
            self.manager.data_dir / "import_uploads" / TEXT_MEDIA_V1_TYPE
        ).resolve()
        if allowed not in path.parents or path.suffix.lower() not in {
            ".tmkb",
            ".tmkbs",
        }:
            raise RuntimeError(f"refusing to clean non-import package: {path}")
        if path.exists():
            await run_blocking(path.unlink)

    @staticmethod
    def _require_valid(report: dict[str, Any]) -> None:
        if (
            str(report.get("integrity") or "") != "ok"
            or int(report.get("foreign_keys") or 0) != 0
        ):
            raise RuntimeError("text media database failed validation")

    async def _activate_stage(
        self,
        *,
        state: dict[str, Any],
        workspace: Path,
        context: JobExecutionContext,
    ) -> None:
        database_id = str(state["database_id"])
        live = Path(state["live"])
        stage = Path(state["stage"])
        previous = Path(state["previous"])
        ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)

        # These filesystem shapes also cover a crash between either rename and
        # the following durable state write.
        if previous.exists() and live.exists() and not stage.exists():
            await self._save_state(workspace, state, phase="activated")
            return

        await context.progress(0.96, "正在原子切换文本媒体知识库代次")
        await self.manager.unload_runtime(ref)
        if not previous.exists():
            if not live.exists():
                raise RuntimeError("live text media database disappeared")
            await run_blocking(live.replace, previous)
        if stage.exists():
            await run_blocking(stage.replace, live)
        if not live.exists():
            raise RuntimeError("staged text media database is unavailable")
        await self._save_state(workspace, state, phase="swapped")
        try:
            detail = await self.manager.library_detail(database_id)
            validation = await (await self.manager.get_runtime(ref)).storage.validate()
            self._require_valid(validation)
            state["activated_provider_id"] = str(detail.get("provider_id") or "")
        except BaseException:
            await self._restore_previous(state, workspace)
            raise
        await self._save_state(workspace, state, phase="activated")

    async def _apply_mutation(
        self,
        *,
        service: TextMediaService,
        kind: str,
        operation: dict[str, Any],
        progress,
    ) -> dict[str, Any]:
        if kind == "text_media_document_ingest":
            path = Path(str(operation["document_path"]))
            return await service.ingest_document(
                filename=str(operation["filename"]),
                title=str(operation.get("title") or ""),
                data=await run_blocking(path.read_bytes),
            )
        if kind == "text_media_ingest_batch":
            return await service.ingest_batch(
                batch_id=str(operation["batch_id"]),
                documents=list(operation["documents"]),
                images=list(operation["images"]),
                chunk_target=int(operation["chunk_target"]),
                chunk_overlap=int(operation["chunk_overlap"]),
                embedding_batch_size=int(operation["embedding_batch_size"]),
                concurrency=int(operation["concurrency"]),
                max_retries=int(operation["max_retries"]),
                media_semantic_calibration_enabled=bool(
                    operation.get("media_semantic_calibration_enabled")
                ),
                progress=progress,
            )
        if kind in {"text_media_entry_create", "text_media_entry_update"}:
            body = await run_blocking(Path(str(operation["body_path"])).read_text, encoding="utf-8")
            if kind == "text_media_entry_create":
                return await service.create_entry(
                    title=str(operation.get("title") or ""), body=body
                )
            return await service.update_entry(
                entry_id=str(operation["entry_id"]),
                title=str(operation.get("title") or ""),
                body=body,
            )
        if kind == "text_media_document_delete":
            document_ids = [str(value) for value in operation["document_ids"]]
            return await service.delete_documents(document_ids)
        if kind == "text_media_entry_delete":
            return await service.delete_entry(str(operation["entry_id"]))
        if kind == "text_media_media_calibration":
            return await service.recalibrate_document_media(
                document_id=str(operation["document_id"]),
                asset_id=str(operation["asset_id"]),
                enabled=bool(operation.get("enabled")),
                media_description=str(operation.get("media_description") or ""),
                progress=progress,
            )
        if kind == "text_media_media_descriptions_update":
            return await service.update_asset_media_descriptions(
                asset_id=str(operation["asset_id"]),
                media_descriptions=[
                    str(value)
                    for value in operation.get("media_descriptions") or []
                ],
                progress=progress,
            )
        raise RuntimeError(f"unsupported text media mutation: {kind}")

    async def _run_mutation(self, context: JobExecutionContext) -> dict[str, Any]:
        job = await context.job()
        kind = str(job.get("kind") or "")
        operation = dict(job.get("operation") or {})
        state, workspace = await self._state(context)
        live = Path(state["live"])
        stage = Path(state["stage"])

        if Path(state["previous"]).exists() and live.exists() and not stage.exists():
            await self._activate_stage(state=state, workspace=workspace, context=context)
            return dict(state.get("result") or {})

        embedding_revision = None
        embedding_context: dict[str, Any] | None = None
        if kind in self.EMBEDDING_MUTATION_KINDS:
            binding = await self._current_embedding_binding(live)
            embedding_revision, embedding_context = (
                await self._long_task_embedding_provider(
                    context=context,
                    state=state,
                    workspace=workspace,
                    provider_id=str(binding.get("provider_id") or ""),
                    expected_revision=int(binding.get("provider_revision") or 0),
                    expected_fingerprint=str(
                        binding.get("provider_fingerprint") or ""
                    ),
                )
            )

        if str(state.get("phase") or "") != "stage_ready":
            await context.progress(0.01, "正在创建文本媒体知识库一致性快照")
            await run_blocking(_copy_library_snapshot, live, stage)
            await self._save_state(workspace, state, phase="snapshot_ready")
            await context.checkpoint(
                {"phase": "snapshot_ready", "saved_at": time.time()},
                progress=0.05,
                message="一致性快照已创建",
            )
            strict_rerank = (
                kind
                == "text_media_media_descriptions_update"
            ) or (
                kind == "text_media_media_calibration"
                and bool(operation.get("enabled"))
            ) or (
                kind == "text_media_ingest_batch"
                and bool(operation.get("media_semantic_calibration_enabled"))
                and any(
                    bool(item.get("document_indexes"))
                    for item in operation.get("images") or []
                )
            )
            service = (
                await self._stage_service(
                    stage,
                    target_provider=build_provider(
                        embedding_revision.config
                    ),
                    strict_rerank=strict_rerank,
                )
                if embedding_revision is not None
                else await self._stage_service_for_current_binding(
                    stage, strict_rerank=strict_rerank
                )
            )
            context_rebuild_required = bool(
                embedding_context
                and embedding_context.get("binding_changed")
            )

            async def mutation_progress(value: float, message: str) -> None:
                base = 0.48 if context_rebuild_required else 0.05
                span = 0.43 if context_rebuild_required else 0.86
                await context.progress(base + span * float(value), message)
                await context.control_point()

            try:
                if context_rebuild_required:
                    async def context_rebuild_progress(
                        value: float, message: str
                    ) -> None:
                        await context.progress(
                            0.05 + 0.40 * float(value), message
                        )
                        await context.control_point()

                    await service.rebuild_embeddings(
                        provider_id=embedding_revision.provider_id,
                        provider_revision=embedding_revision.revision,
                        provider_fingerprint=(
                            embedding_revision.config_sha256
                        ),
                        progress=context_rebuild_progress,
                    )
                result = await self._apply_mutation(
                    service=service,
                    kind=kind,
                    operation=operation,
                    progress=mutation_progress,
                )
                validation = await service.storage.validate()
                self._require_valid(validation)
            finally:
                await service.close()
            if embedding_context is not None:
                result = {
                    **dict(result),
                    "embedding_context": embedding_context,
                }
            await self._save_state(
                workspace,
                state,
                phase="stage_ready",
                result=result,
                embedding_context_applied=True,
            )
            await context.checkpoint(
                {"phase": "stage_ready", "saved_at": time.time()},
                progress=0.94,
                message="新 generation 与索引已完成并通过校验",
            )

        if embedding_context is not None:
            latest = await self.manager.control.get_provider(
                str(embedding_context["provider_id"])
            )
            if (
                latest is None
                or latest.config_sha256
                != str(embedding_context["provider_fingerprint"])
            ):
                raise ValueError(
                    "Embedding Provider changed before text media task "
                    "activation; the staged database was not activated"
                )

        await self._activate_stage(state=state, workspace=workspace, context=context)
        await context.progress(1.0, "文本媒体知识库任务已完成")
        return {
            **dict(state.get("result") or {}),
            "database_type": TEXT_MEDIA_V1_TYPE,
            "database_id": str(state["database_id"]),
            "task_kind": kind,
        }

    async def _run_index_rebuild(
        self, context: JobExecutionContext
    ) -> dict[str, Any]:
        job = await context.job()
        operation = dict(job.get("operation") or {})
        state, workspace = await self._state(context)
        database_id = str(state["database_id"])
        live = Path(state["live"])
        stage = Path(state["stage"])
        previous = Path(state["previous"])
        provider_id = str(operation.get("provider_id") or "").strip()
        if not provider_id:
            current = await self.manager.library_detail(database_id)
            provider_id = str(current.get("provider_id") or "")
        revision, embedding_context = (
            await self._long_task_embedding_provider(
                context=context,
                state=state,
                workspace=workspace,
                provider_id=provider_id,
            )
        )
        provider_id = revision.provider_id
        provider_revision = revision.revision
        provider_fingerprint = revision.config_sha256
        phase = str(state.get("phase") or "")
        if phase == "stage_ready" or (
            previous.exists() and live.exists() and not stage.exists()
        ):
            provider_id = str(state.get("provider_id") or provider_id)
            provider_revision = int(state.get("provider_revision") or 0)
            provider_fingerprint = str(
                state.get("provider_fingerprint") or ""
            )
            revision = await self.manager.control.get_provider(provider_id)
            if (
                revision is None
                or not revision.config.enabled
                or provider_kind(revision.config.type) != "embedding"
                or revision.config_sha256 != provider_fingerprint
            ):
                raise ValueError(
                    "target Embedding Provider changed while the rebuild was "
                    "running; the staged database was not activated"
                )
        else:
            revision = await self.manager.control.get_provider(provider_id)
            if (
                revision is None
                or not revision.config.enabled
                or provider_kind(revision.config.type) != "embedding"
            ):
                raise ValueError("target Embedding Provider is unavailable")
            provider_revision = revision.revision
            provider_fingerprint = revision.config_sha256

        if phase != "stage_ready":
            # A crash can occur after the staged directory became live but
            # before the state write. The presence of `previous` and absence
            # of `stage` proves that the two-directory switch completed.
            if previous.exists() and live.exists() and not stage.exists():
                await self._activate_stage(
                    state=state, workspace=workspace, context=context
                )
                detail = await self.manager.library_detail(database_id)
                if str(detail.get("provider_id") or "") != revision.provider_id:
                    raise RuntimeError("provider binding did not activate")
                return {
                    **dict(state.get("result") or {}),
                    "database_type": TEXT_MEDIA_V1_TYPE,
                    "database_id": database_id,
                    "provider_id": revision.provider_id,
                    "provider_revision": provider_revision,
                    "provider_fingerprint": provider_fingerprint,
                    "embedding_context": embedding_context,
                    "reason": str(operation.get("reason") or "manual"),
                }
            await context.progress(0.01, "正在创建文本媒体知识库一致性快照")
            await run_blocking(_copy_library_snapshot, live, stage)
            await self._save_state(workspace, state, phase="snapshot_ready")
            await context.checkpoint(
                {
                    "phase": "snapshot_ready",
                    "saved_at": time.time(),
                },
                progress=0.05,
                message="一致性快照已创建",
            )
            target_provider = build_provider(revision.config)
            service = await self._stage_service(
                stage,
                target_provider=target_provider,
            )
            async def stage_progress(value: float, message: str) -> None:
                await context.progress(0.05 + 0.86 * float(value), message)
                await context.control_point()

            try:
                result = await service.rebuild_embeddings(
                    provider_id=revision.provider_id,
                    provider_revision=provider_revision,
                    provider_fingerprint=provider_fingerprint,
                    progress=stage_progress,
                )
            finally:
                await service.close()
            await self._save_state(
                workspace,
                state,
                phase="stage_ready",
                result=result,
                provider_id=revision.provider_id,
                provider_revision=provider_revision,
                provider_fingerprint=provider_fingerprint,
            )
            await context.checkpoint(
                {"phase": "stage_ready", "saved_at": time.time()},
                progress=0.94,
                message="新 Provider 向量、媒体校准与索引已完成",
            )

        latest = await self.manager.control.get_provider(provider_id)
        if (
            latest is None
            or not latest.config.enabled
            or provider_kind(latest.config.type) != "embedding"
            or latest.config_sha256 != provider_fingerprint
        ):
            raise ValueError(
                "target Embedding Provider changed before activation; the "
                "staged database was not activated"
            )
        await self._activate_stage(state=state, workspace=workspace, context=context)
        detail = await self.manager.library_detail(database_id)
        if str(detail.get("provider_id") or "") != revision.provider_id:
            await self._restore_previous(state, workspace)
            raise RuntimeError("provider binding did not activate")
        await context.progress(1.0, "文本媒体知识库 Provider 切换已完成")
        return {
            **dict(state.get("result") or {}),
            "database_type": TEXT_MEDIA_V1_TYPE,
            "database_id": database_id,
            "provider_id": revision.provider_id,
            "provider_revision": provider_revision,
            "provider_fingerprint": provider_fingerprint,
            "embedding_context": embedding_context,
            "reason": str(operation.get("reason") or "manual"),
        }

    async def _import_state(
        self, context: JobExecutionContext
    ) -> tuple[dict[str, Any], Path, dict[str, Any]]:
        job = await context.job()
        operation = dict(job.get("operation") or {})
        workspace = self.workspace(job)
        state_path = workspace / "state.json"
        if state_path.exists():
            return (
                json.loads(state_path.read_text(encoding="utf-8")),
                workspace,
                operation,
            )
        workspace.mkdir(parents=True, exist_ok=True)
        kind = str(job.get("kind") or "")
        if kind == "tmkb_import":
            targets = [str(operation.get("target_id") or "")]
        elif kind == "tmkbs_import":
            targets = [
                str(item.get("target_id") or "")
                for item in list(operation.get("items") or [])
            ]
        else:
            raise RuntimeError(f"unsupported text media import task: {kind}")
        if not targets or any(not value for value in targets):
            raise ValueError("text media import task is missing a target ID")
        if len(set(targets)) != len(targets):
            raise ValueError("text media import task contains duplicate target IDs")
        state = {
            "version": 1,
            "kind": kind,
            "phase": "created",
            "targets": targets,
            "moved": [],
            "created_at": time.time(),
        }
        await run_blocking(atomic_write_json, state_path, state)
        return state, workspace, operation

    def _import_final(self, target_id: str) -> Path:
        return database_type_registry.data_dir(
            self.manager.data_dir,
            DatabaseRef(TEXT_MEDIA_V1_TYPE, target_id),
        )

    def _owner_marker(self, root: Path) -> Path:
        return root / self.INSTALL_OWNER_FILENAME

    def _is_owned_install(self, root: Path, job_id: str) -> bool:
        marker = self._owner_marker(root)
        if not marker.is_file():
            return False
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return str(payload.get("job_id") or "") == job_id

    async def _mark_owned_install(self, root: Path, job_id: str) -> None:
        await run_blocking(
            atomic_write_json,
            self._owner_marker(root),
            {"job_id": job_id, "created_at": time.time()},
        )

    async def _prepare_single_import(
        self,
        *,
        context: JobExecutionContext,
        state: dict[str, Any],
        workspace: Path,
        operation: dict[str, Any],
    ) -> None:
        target_id = str(operation["target_id"])
        ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, target_id)
        final = self._import_final(target_id)
        if await self.manager.control.database_identity(ref) or final.exists():
            raise ValueError(f"target library ID already exists: {target_id}")
        prepared_root = workspace / "prepared" / target_id
        if prepared_root.parent.exists():
            await self._cleanup_path(prepared_root.parent)
        prepared_root.mkdir(parents=True, exist_ok=False)
        await context.progress(0.05, "validating and preparing knowledge library")
        prepared = await prepare_tmkb_install(
            manager=self.manager,
            package_path=Path(str(operation["package_path"])),
            target_id=target_id,
            name_override=str(operation.get("name") or "").strip() or None,
            staging=prepared_root,
        )
        await self._mark_owned_install(prepared.root, context.job_id)
        await self._save_state(
            workspace,
            state,
            phase="stage_ready",
            prepared={target_id: str(prepared.root)},
        )
        await context.checkpoint(
            {"phase": "stage_ready", "saved_at": time.time()},
            progress=0.82,
            message="knowledge library package is ready for atomic activation",
        )

    async def _prepare_batch_import(
        self,
        *,
        context: JobExecutionContext,
        state: dict[str, Any],
        workspace: Path,
        operation: dict[str, Any],
    ) -> None:
        items = list(operation.get("items") or [])
        targets = [str(item["target_id"]) for item in items]
        for target_id in targets:
            ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, target_id)
            if (
                await self.manager.control.database_identity(ref)
                or self._import_final(target_id).exists()
            ):
                raise ValueError(f"target library ID already exists: {target_id}")
        package_root = workspace / "package"
        prepared_parent = workspace / "prepared"
        if package_root.exists():
            await self._cleanup_path(package_root)
        if prepared_parent.exists():
            await self._cleanup_path(prepared_parent)
        await context.progress(0.03, "validating knowledge library batch package")
        manifest = await run_blocking(
            extract_tmkbs,
            Path(str(operation["package_path"])),
            package_root,
        )
        available = {
            str(item["database_id"]): item for item in manifest["libraries"]
        }
        prepared_paths: dict[str, str] = {}
        for index, item in enumerate(items):
            source_id = str(item.get("source_id") or "")
            target_id = str(item["target_id"])
            child = available.get(source_id)
            if child is None:
                raise ValueError(
                    f"source library is not present in package: {source_id}"
                )
            staging = prepared_parent / target_id
            staging.mkdir(parents=True, exist_ok=False)
            prepared = await prepare_tmkb_install(
                manager=self.manager,
                package_path=package_root.joinpath(
                    *Path(str(child["member"])).parts
                ),
                target_id=target_id,
                name_override=str(item.get("name") or "").strip() or None,
                staging=staging,
            )
            await self._mark_owned_install(prepared.root, context.job_id)
            prepared_paths[target_id] = str(prepared.root)
            await context.progress(
                0.08 + 0.68 * ((index + 1) / len(items)),
                f"prepared {index + 1}/{len(items)} knowledge libraries",
            )
            await context.control_point()
        await self._save_state(
            workspace,
            state,
            phase="stage_ready",
            prepared=prepared_paths,
        )
        await context.checkpoint(
            {"phase": "stage_ready", "saved_at": time.time()},
            progress=0.78,
            message="knowledge library batch is ready for atomic activation",
        )

    async def _activate_import(
        self,
        *,
        context: JobExecutionContext,
        state: dict[str, Any],
        workspace: Path,
    ) -> list[dict[str, Any]]:
        targets = [str(value) for value in state["targets"]]
        moved = [str(value) for value in state.get("moved") or []]
        prepared = dict(state.get("prepared") or {})
        for index, target_id in enumerate(targets):
            final = self._import_final(target_id)
            if final.exists():
                if not self._is_owned_install(final, context.job_id):
                    raise RuntimeError(
                        f"refusing to replace an unowned library: {target_id}"
                    )
            else:
                source = Path(str(prepared.get(target_id) or ""))
                if not source.is_dir() or not self._is_owned_install(
                    source, context.job_id
                ):
                    raise RuntimeError(
                        f"prepared knowledge library is unavailable: {target_id}"
                    )
                final.parent.mkdir(parents=True, exist_ok=True)
                await run_blocking(source.replace, final)
            if target_id not in moved:
                moved.append(target_id)
            await self._save_state(
                workspace,
                state,
                phase="moving",
                moved=moved,
            )
            await context.progress(
                0.80 + 0.12 * ((index + 1) / len(targets)),
                f"activated {index + 1}/{len(targets)} knowledge libraries",
            )
            await context.control_point()

        refs = [DatabaseRef(TEXT_MEDIA_V1_TYPE, value) for value in targets]
        identities = [
            await self.manager.control.database_identity(ref) for ref in refs
        ]
        if any(identities) and not all(identities):
            raise RuntimeError("batch import identity registration is incomplete")
        if not any(identities):
            await self.manager.control.register_database_identities(
                refs,
                category=DATABASE_CATEGORY_KNOWLEDGE,
            )
        await self._save_state(workspace, state, phase="installed", moved=moved)
        results = [await self.manager.library_detail(value) for value in targets]
        for target_id in targets:
            marker = self._owner_marker(self._import_final(target_id))
            if marker.exists():
                await run_blocking(marker.unlink)
        return results

    async def _run_import(self, context: JobExecutionContext) -> dict[str, Any]:
        state, workspace, operation = await self._import_state(context)
        phase = str(state.get("phase") or "")
        if phase == "installed":
            results = [
                await self.manager.library_detail(str(value))
                for value in state["targets"]
            ]
        else:
            if phase not in {"stage_ready", "moving"}:
                if str(state["kind"]) == "tmkb_import":
                    await self._prepare_single_import(
                        context=context,
                        state=state,
                        workspace=workspace,
                        operation=operation,
                    )
                else:
                    await self._prepare_batch_import(
                        context=context,
                        state=state,
                        workspace=workspace,
                        operation=operation,
                    )
            results = await self._activate_import(
                context=context,
                state=state,
                workspace=workspace,
            )
        await context.progress(1.0, "knowledge library import completed")
        if str(state["kind"]) == "tmkb_import":
            return results[0]
        return {"libraries": results, "count": len(results)}

    async def _rollback_import(self, context: JobExecutionContext) -> None:
        state, workspace, operation = await self._import_state(context)
        moved = {str(value) for value in state.get("moved") or []}
        for target_id in reversed([str(value) for value in state["targets"]]):
            ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, target_id)
            final = self._import_final(target_id)
            owned = target_id in moved or (
                final.exists() and self._is_owned_install(final, context.job_id)
            )
            if not owned:
                continue
            await self.manager.unload_runtime(ref)
            if await self.manager.control.database_identity(ref):
                await self.manager.control.delete_database_identity(ref)
            if final.exists():
                type_root = database_type_registry.type_root(
                    self.manager.data_dir, TEXT_MEDIA_V1_TYPE
                ).resolve()
                resolved = final.resolve()
                if resolved == type_root or type_root not in resolved.parents:
                    raise RuntimeError(
                        f"refusing to remove import path outside type root: {final}"
                    )
                await run_blocking(shutil.rmtree, final, True)
        await self._cleanup_import_package(operation)
        await self._cleanup_path(workspace)

    async def _cancel_queued_import(
        self, context: JobExecutionContext
    ) -> None:
        job = await context.job()
        await self._cleanup_import_package(dict(job.get("operation") or {}))
        workspace = self.workspace(job)
        if workspace.exists():
            await self._cleanup_path(workspace)

    async def _finalize_import(self, context: JobExecutionContext) -> None:
        job = await context.job()
        await self._cleanup_import_package(dict(job.get("operation") or {}))
        workspace = self.workspace(job)
        if workspace.exists():
            await self._cleanup_path(workspace)

    async def _restore_previous(
        self, state: dict[str, Any], workspace: Path
    ) -> None:
        database_id = str(state["database_id"])
        ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
        live = Path(state["live"])
        previous = Path(state["previous"])
        stage = Path(state["stage"])
        await self.manager.unload_runtime(ref)
        failed = workspace / "failed"
        if previous.exists():
            if failed.exists():
                await run_blocking(shutil.rmtree, failed, True)
            if live.exists():
                await run_blocking(live.replace, failed)
            await run_blocking(previous.replace, live)
        if stage.exists():
            await run_blocking(shutil.rmtree, stage, True)
        if failed.exists():
            await run_blocking(shutil.rmtree, failed, True)
        if live.exists():
            await self.manager.get_runtime(ref)
        await self._save_state(workspace, state, phase="rolled_back")

    async def _rollback(self, context: JobExecutionContext) -> None:
        job = await context.job()
        state, workspace = await self._state(context)
        await self._restore_previous(state, workspace)
        await self._cleanup_path(self._operation_input_root(job))
        await self._cleanup_path(workspace)

    async def _cancel_queued(self, context: JobExecutionContext) -> None:
        job = await context.job()
        workspace = self.workspace(job)
        await self._cleanup_path(workspace)
        await self._cleanup_path(self._operation_input_root(job))

    async def _finalize_completed(self, context: JobExecutionContext) -> None:
        job = await context.job()
        workspace = self.workspace(job)
        await self._cleanup_path(workspace)
        await self._cleanup_path(self._operation_input_root(job))
