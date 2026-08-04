from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import httpx


CHARACTER_MEDIA = (
    (
        "澄月全身立绘.png",
        "澄月身穿狱狼龙贝塔套装、手持狱牙刀的全身战斗服立绘设定图",
    ),
    (
        "澄月的立绘头像.png",
        "澄月白色短发、黑色刘海、红色眼睛和黑红色角状发饰的脸部特写立绘头像",
    ),
    (
        "澄月战斗场景插图.jpg",
        "澄月身穿狱狼龙贝塔套装、手持缠绕暗红雷电的狱牙刀，在蓝黑色竞技场战斗的场景插图",
    ),
)
MEME_IMAGES = (
    "“原来是劣等模型”claude版表情包.jpg",
    "“原来是劣等模型”deepseek版表情包.jpg",
    "“原来是劣等模型”gemini版表情包.jpg",
    "“原来是劣等模型”gpt版表情包.jpg",
)
TEXT_ONLY_DOCUMENTS = (
    "澄月的斩魄刀的设定.md",
    "怪物猎人冰原终盘太刀配装.txt",
    "群友.txt",
    "《怪物猎人世界：冰原》全特殊装备获取与强化深度解析.docx",
    "闪光的哈萨维 - 富野由悠季 - 20100824.pdf",
)


def request_json(
    client: httpx.Client, method: str, url: str, **kwargs: Any
) -> dict[str, Any]:
    response = client.request(method, url, **kwargs)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError(f"unexpected response from {url}")
    return value


def wait_job(
    client: httpx.Client,
    base_url: str,
    job_id: str,
    *,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = request_json(client, "GET", f"{base_url}/api/v1/jobs/{job_id}")
        status = str(job.get("status") or "")
        if status == "completed":
            return job
        if status in {"failed", "stopped", "cancelled"}:
            raise RuntimeError(str(job.get("error") or job.get("message") or job))
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} did not finish in {timeout:.0f}s")


def media_type(path: Path) -> str:
    suffix = path.suffix.casefold()
    return {
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".txt": "text/plain",
        ".pdf": "application/pdf",
        ".docx": (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")


def post_batch(
    client: httpx.Client,
    base_url: str,
    library_url: str,
    *,
    documents: list[Path],
    images: list[tuple[Path, list[int], str]],
    calibrate: bool,
) -> dict[str, Any]:
    manifest = {
        "chunk_target": 1200,
        "chunk_overlap": 150,
        "embedding_batch_size": 32,
        "concurrency": 3,
        "max_retries": 3,
        "media_semantic_calibration_enabled": calibrate,
        "images": [
            {
                "document_indexes": indexes,
                "media_description": description,
            }
            for _, indexes, description in images
        ],
    }
    with ExitStack() as stack:
        files: list[tuple[str, tuple[str, Any, str]]] = []
        for path in documents:
            files.append(
                (
                    "documents[]",
                    (path.name, stack.enter_context(path.open("rb")), media_type(path)),
                )
            )
        for path, _, _ in images:
            files.append(
                (
                    "images[]",
                    (path.name, stack.enter_context(path.open("rb")), media_type(path)),
                )
            )
        started = request_json(
            client,
            "POST",
            f"{library_url}/ingest-batches",
            data={"manifest": json.dumps(manifest, ensure_ascii=False)},
            files=files,
        )
    return wait_job(client, base_url, str(started["job_id"]))


def reset_library(client: httpx.Client, base_url: str, library_url: str) -> None:
    documents = request_json(client, "GET", f"{library_url}/documents?limit=200")
    document_ids = [str(item["id"]) for item in documents.get("items", [])]
    if document_ids:
        started = request_json(
            client,
            "POST",
            f"{library_url}/documents/batch-delete",
            json={"document_ids": document_ids},
        )
        wait_job(client, base_url, str(started["job_id"]))
    assets = request_json(client, "GET", f"{library_url}/assets")
    for item in assets.get("items", []):
        request_json(
            client,
            "DELETE",
            f"{library_url}/assets/{item['id']}",
        )
    remaining_documents = request_json(
        client, "GET", f"{library_url}/documents?limit=1"
    )
    remaining_assets = request_json(client, "GET", f"{library_url}/assets")
    if int(remaining_documents.get("total") or 0) or remaining_assets.get("items"):
        raise RuntimeError("library reset did not remove all documents and images")


def library_snapshot(client: httpx.Client, library_url: str) -> dict[str, Any]:
    documents = request_json(
        client, "GET", f"{library_url}/documents?limit=200&sort=title_asc"
    )
    document_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    for summary in documents.get("items", []):
        detail = request_json(client, "GET", f"{library_url}/documents/{summary['id']}")
        source_name = str(detail["original_name"])
        document_rows.append(
            {
                "source": source_name,
                "source_sha256": str(detail["source_sha256"]),
                "content_sha256": str(detail["content_sha256"]),
                "normalized_content_sha256": hashlib.sha256(
                    str(detail["content"]).encode("utf-8")
                ).hexdigest(),
                "parser_id": str(detail["parser_id"]),
                "chunk_count": int(detail["chunk_count"]),
                "media": sorted(
                    (
                        str(item["original_name"]),
                        str(item.get("media_description") or ""),
                        str(item.get("output_policy") or ""),
                    )
                    for item in detail.get("associated_media", [])
                ),
            }
        )
        offset = 0
        while True:
            page = request_json(
                client,
                "GET",
                (
                    f"{library_url}/chunks?document_id={summary['id']}"
                    f"&offset={offset}&limit=200&sort=ordinal_asc"
                ),
            )
            for item in page.get("items", []):
                chunk_rows.append(
                    {
                        "source": source_name,
                        "ordinal": int(item["ordinal"]),
                        "content_sha256": str(item["content_sha256"]),
                        "text_sha256": hashlib.sha256(
                            str(item["text"]).encode("utf-8")
                        ).hexdigest(),
                        "media_count": int(item.get("media_count") or 0),
                    }
                )
            offset += len(page.get("items", []))
            if offset >= int(page.get("total") or 0):
                break
    assets = request_json(client, "GET", f"{library_url}/assets")
    asset_rows = sorted(
        (
            {
                "name": str(item["original_name"]),
                "sha256": str(item["sha256"]),
                "description": str(item.get("media_description") or ""),
                "description_source": str(item.get("description_source") or ""),
                "vector_status": str(item.get("media_vector_status") or ""),
            }
            for item in assets.get("items", [])
        ),
        key=lambda item: item["name"],
    )
    snapshot = {
        "documents": sorted(document_rows, key=lambda item: item["source"]),
        "chunks": sorted(
            chunk_rows, key=lambda item: (item["source"], item["ordinal"])
        ),
        "assets": asset_rows,
    }
    snapshot["sha256"] = hashlib.sha256(
        json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the text_media_v1 standard/media-only benchmark via the "
            "protected formal API."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--api-key", required=True)
    parser.add_argument(
        "--libraries",
        nargs="+",
        default=["beileite_test", "beileite_test2"],
    )
    parser.add_argument(
        "--material-root", type=Path, default=Path(r"D:\git_test\RAG\素材")
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--confirm-libraries",
        nargs="*",
        default=[],
        help="Must contain exactly the requested library IDs when --reset is used.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    library_ids = list(dict.fromkeys(str(value) for value in args.libraries))
    if args.reset and set(args.confirm_libraries) != set(library_ids):
        parser.error("--reset requires --confirm-libraries for every requested ID")
    if not args.reset:
        parser.error("this benchmark rebuild requires the guarded --reset flag")

    documents_root = args.material_root / "文档"
    images_root = args.material_root / "图片"
    first_document = documents_root / "澄月.md"
    ecology_document = documents_root / "怪物猎人冰原设定集生态翻译17——狱狼龙.txt"
    ecology_image = images_root / "狱狼龙插图.jpg"
    text_only_paths = [documents_root / name for name in TEXT_ONLY_DOCUMENTS]
    character_images = [
        (images_root / name, [0], description) for name, description in CHARACTER_MEDIA
    ]
    meme_images = [(images_root / name, [], "") for name in MEME_IMAGES]
    required = [
        first_document,
        ecology_document,
        ecology_image,
        *text_only_paths,
        *(item[0] for item in character_images),
        *(item[0] for item in meme_images),
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    base_url = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.api_key}"}
    report: dict[str, Any] = {
        "created_at": time.time(),
        "libraries": {},
    }
    with httpx.Client(headers=headers, timeout=180.0) as client:
        before: dict[str, dict[str, Any]] = {}
        for library_id in library_ids:
            library_url = (
                f"{base_url}/api/v1/knowledge-libraries/text_media_v1/{library_id}"
            )
            detail = request_json(client, "GET", library_url)
            if (
                detail.get("id") != library_id
                or detail.get("database_type") != "text_media_v1"
            ):
                raise RuntimeError(f"library identity mismatch: {library_id}")
            before[library_id] = detail
        non_rerank_settings = {
            library_id: {
                key: value
                for key, value in detail["retrieval_settings"].items()
                if not key.startswith("rerank_")
            }
            for library_id, detail in before.items()
        }
        if (
            len(
                {
                    json.dumps(value, sort_keys=True)
                    for value in non_rerank_settings.values()
                }
            )
            != 1
        ):
            raise RuntimeError("benchmark libraries have different non-Rerank settings")

        for library_id in library_ids:
            library_url = (
                f"{base_url}/api/v1/knowledge-libraries/text_media_v1/{library_id}"
            )
            reset_library(client, base_url, library_url)
            jobs = [
                post_batch(
                    client,
                    base_url,
                    library_url,
                    documents=[first_document],
                    images=character_images,
                    calibrate=True,
                ),
                post_batch(
                    client,
                    base_url,
                    library_url,
                    documents=[ecology_document],
                    images=[(ecology_image, [0], "")],
                    calibrate=True,
                ),
                post_batch(
                    client,
                    base_url,
                    library_url,
                    documents=text_only_paths,
                    images=[],
                    calibrate=False,
                ),
                post_batch(
                    client,
                    base_url,
                    library_url,
                    documents=[],
                    images=meme_images,
                    calibrate=False,
                ),
            ]
            detail = request_json(client, "GET", library_url)
            stats = detail.get("stats") or {}
            if (
                int(stats.get("documents") or 0) != 7
                or int(stats.get("images") or 0) != 8
            ):
                raise RuntimeError(
                    f"unexpected rebuilt statistics for {library_id}: {stats}"
                )
            snapshot = library_snapshot(client, library_url)
            report["libraries"][library_id] = {
                "provider_id": detail.get("provider_id"),
                "provider_revision": detail.get("provider_revision"),
                "rerank_provider_id": detail.get("rerank_provider_id"),
                "stats": stats,
                "job_ids": [str(item["id"]) for item in jobs],
                "snapshot": snapshot,
            }
        digests = {
            value["snapshot"]["sha256"] for value in report["libraries"].values()
        }
        if len(digests) != 1:
            raise RuntimeError("rebuilt benchmark libraries are not content-identical")
        report["content_snapshot_sha256"] = next(iter(digests))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "libraries": {
                    key: {
                        "stats": value["stats"],
                        "rerank_provider_id": value["rerank_provider_id"],
                    }
                    for key, value in report["libraries"].items()
                },
                "content_snapshot_sha256": report["content_snapshot_sha256"],
                "report": str(args.report) if args.report else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
