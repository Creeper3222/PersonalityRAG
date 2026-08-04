from __future__ import annotations

import json
from pathlib import Path

import pytest

from personalityrag.library_types.text_media_v1.storage import TextMediaStorage
from personalityrag.library_types.text_media_v1.visual_intent_policy import (
    DEFAULT_VISUAL_INTENT_POLICY,
    DEFAULT_VISUAL_INTENT_POLICY_PATH,
    MAX_VISUAL_INTENT_POLICY_CSV_BYTES,
    VISUAL_INTENT_POLICY_FILENAME,
    VISUAL_INTENT_POLICY_KEYS,
    migrate_legacy_visual_intent_policy,
    normalize_visual_intent_policy,
    parse_visual_intent_policy_csv,
    read_effective_visual_intent_policy,
    serialize_visual_intent_policy_csv,
    write_visual_intent_policy_override,
)


def test_visual_intent_csv_complete_and_subset_roundtrip() -> None:
    policy = {
        "visual_object_terms": ["角色,立绘", '带"引号"的头像', "澄月"],
        "lookup_action_terms": [],
        "generation_action_terms": ["绘制", "DRAW"],
        "reference_connector_terms": ["根据"],
    }
    payload = serialize_visual_intent_policy_csv(policy)

    assert payload.startswith(b"\xef\xbb\xbf")
    restored, categories = parse_visual_intent_policy_csv(
        payload,
        require_complete=True,
    )
    assert categories == VISUAL_INTENT_POLICY_KEYS
    assert restored == {
        "visual_object_terms": ["角色,立绘", '带"引号"的头像', "澄月"],
        "lookup_action_terms": [],
        "generation_action_terms": ["绘制", "draw"],
        "reference_connector_terms": ["根据"],
    }

    subset = serialize_visual_intent_policy_csv(
        restored,
        categories=["lookup_action_terms", "reference_connector_terms"],
    )
    subset_policy, subset_categories = parse_visual_intent_policy_csv(subset)
    assert subset_categories == (
        "lookup_action_terms",
        "reference_connector_terms",
    )
    assert subset_policy == {
        "lookup_action_terms": [],
        "reference_connector_terms": ["根据"],
    }


def test_visual_intent_csv_normalizes_deduplicates_and_preserves_order() -> None:
    payload = (
        "visual_object_terms\r\n"
        "  Portrait  \r\n"
        "portrait\r\n"
        "ＰＯＲＴＲＡＩＴ\r\n"
        "头像\r\n"
    ).encode("utf-8-sig")

    policy, categories = parse_visual_intent_policy_csv(payload)

    assert categories == ("visual_object_terms",)
    assert policy == {"visual_object_terms": ["portrait", "头像"]}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            b"visual_object_terms,visual_object_terms\n",
            "duplicate",
        ),
        (b"visual_object_terms,unknown_terms\n", "unknown"),
        (
            b"visual_object_terms\nvisual_object_terms\n",
            "repeats its header",
        ),
        (b"visual_object_terms,lookup_action_terms\nportrait\n", "fields"),
    ],
)
def test_visual_intent_csv_rejects_invalid_headers_and_rows(
    payload: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_visual_intent_policy_csv(payload)


def test_visual_intent_csv_rejects_encoding_size_and_term_limits() -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        parse_visual_intent_policy_csv(b"visual_object_terms\n\xff\n")
    with pytest.raises(ValueError, match="1 MiB"):
        parse_visual_intent_policy_csv(
            b"x" * (MAX_VISUAL_INTENT_POLICY_CSV_BYTES + 1)
        )
    too_many = (
        "visual_object_terms\n"
        + "\n".join(f"term-{index}" for index in range(257))
        + "\n"
    ).encode("utf-8-sig")
    with pytest.raises(ValueError, match="at most 256"):
        parse_visual_intent_policy_csv(too_many)


def test_type_default_csv_is_bom_complete_and_read_only_resource() -> None:
    payload = DEFAULT_VISUAL_INTENT_POLICY_PATH.read_bytes()
    policy, categories = parse_visual_intent_policy_csv(
        payload,
        require_complete=True,
    )

    assert payload.startswith(b"\xef\xbb\xbf")
    assert categories == VISUAL_INTENT_POLICY_KEYS
    assert policy == normalize_visual_intent_policy(DEFAULT_VISUAL_INTENT_POLICY)
    assert DEFAULT_VISUAL_INTENT_POLICY_PATH.name == (
        "visual_intent_policy.default.csv"
    )


def test_single_library_override_csv_and_default_inheritance(tmp_path: Path) -> None:
    defaults = normalize_visual_intent_policy(DEFAULT_VISUAL_INTENT_POLICY)
    custom = normalize_visual_intent_policy(defaults)
    custom["generation_action_terms"].append("创作")

    assert write_visual_intent_policy_override(tmp_path, custom) is True
    target = tmp_path / VISUAL_INTENT_POLICY_FILENAME
    assert target.read_bytes().startswith(b"\xef\xbb\xbf")
    assert read_effective_visual_intent_policy(tmp_path) == custom

    assert write_visual_intent_policy_override(tmp_path, defaults) is False
    assert not target.exists()
    assert read_effective_visual_intent_policy(tmp_path) == defaults


async def _create_storage(root: Path, database_id: str) -> TextMediaStorage:
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id=database_id,
        name=database_id,
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    return storage


async def _write_legacy_policy(
    storage: TextMediaStorage,
    policy: dict[str, list[str]],
) -> None:
    db = await storage.pool.acquire()
    try:
        await db.execute(
            "UPDATE library_meta SET retrieval_config_json=? WHERE singleton=1",
            (json.dumps({"rrf_k": 77, "visual_intent_policy": policy}),),
        )
        await db.commit()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_legacy_json_policy_migrates_atomically_to_csv(tmp_path: Path) -> None:
    root = tmp_path / "custom"
    storage = await _create_storage(root, "custom")
    custom = normalize_visual_intent_policy(DEFAULT_VISUAL_INTENT_POLICY)
    custom["generation_action_terms"].append("创作")
    try:
        await _write_legacy_policy(storage, custom)

        effective = await migrate_legacy_visual_intent_policy(storage, root)

        assert effective == custom
        assert (root / VISUAL_INTENT_POLICY_FILENAME).exists()
        stored = json.loads((await storage.metadata())["retrieval_config_json"])
        assert stored["rrf_k"] == 77
        assert "visual_intent_policy" not in stored
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_legacy_default_policy_removes_json_without_override(
    tmp_path: Path,
) -> None:
    root = tmp_path / "default"
    storage = await _create_storage(root, "default")
    defaults = normalize_visual_intent_policy(DEFAULT_VISUAL_INTENT_POLICY)
    try:
        await _write_legacy_policy(storage, defaults)

        effective = await migrate_legacy_visual_intent_policy(storage, root)

        assert effective == defaults
        assert not (root / VISUAL_INTENT_POLICY_FILENAME).exists()
        stored = json.loads((await storage.metadata())["retrieval_config_json"])
        assert "visual_intent_policy" not in stored
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_existing_csv_wins_and_corrupt_csv_never_falls_back(
    tmp_path: Path,
) -> None:
    root = tmp_path / "csv-wins"
    storage = await _create_storage(root, "csv-wins")
    legacy = normalize_visual_intent_policy(DEFAULT_VISUAL_INTENT_POLICY)
    legacy["generation_action_terms"].append("旧词")
    custom = normalize_visual_intent_policy(DEFAULT_VISUAL_INTENT_POLICY)
    custom["generation_action_terms"].append("csv词")
    try:
        await _write_legacy_policy(storage, legacy)
        write_visual_intent_policy_override(root, custom)
        assert await migrate_legacy_visual_intent_policy(storage, root) == custom

        (root / VISUAL_INTENT_POLICY_FILENAME).write_bytes(
            b"unknown_column\nvalue\n"
        )
        with pytest.raises(ValueError, match="unknown"):
            await migrate_legacy_visual_intent_policy(storage, root)
    finally:
        await storage.close()
