from __future__ import annotations

import pytest
from pydantic import ValidationError

from personalityrag.config import ProviderConfig
from personalityrag.control import ControlStore
from personalityrag.identifiers import validate_identifier
from personalityrag.schemas import LibraryCreate, LibraryUpdate, ProviderCopy, ProviderCreate, ProviderUpdate


@pytest.mark.parametrize("identifier", ["Astrbot", "beileite_test", "provider-01", "A1_b-2"])
def test_identifier_accepts_ascii_letters_digits_underscores_and_hyphens(identifier: str):
    assert validate_identifier(identifier) == identifier


@pytest.mark.parametrize("identifier", ["", "space id", " id", "id ", "id.name", "记忆库", "id\nnext"])
def test_identifier_rejects_whitespace_unicode_and_other_punctuation(identifier: str):
    with pytest.raises(ValueError, match=r"\[a-zA-Z0-9_-\]"):
        validate_identifier(identifier)


@pytest.mark.parametrize("identifier", ["bad id", "provider.id", "提供商"])
def test_public_write_schemas_reject_invalid_library_and_provider_ids(identifier: str):
    with pytest.raises(ValidationError, match=r"\[a-zA-Z0-9_-\]"):
        LibraryCreate(id=identifier, name="Nickname can use 中文", provider_id="seed_provider")
    with pytest.raises(ValidationError, match=r"\[a-zA-Z0-9_-\]"):
        LibraryUpdate(id=identifier)
    with pytest.raises(ValidationError, match=r"\[a-zA-Z0-9_-\]"):
        ProviderCreate(
            id=identifier,
            display_name="Nickname can use spaces 和中文",
            type="vllm_embedding",
            api_base="http://127.0.0.1:8001/v1",
            model="fixture-model",
        )
    with pytest.raises(ValidationError, match=r"\[a-zA-Z0-9_-\]"):
        ProviderUpdate(id=identifier)
    with pytest.raises(ValidationError, match=r"\[a-zA-Z0-9_-\]"):
        ProviderCopy(new_id=identifier)


@pytest.mark.asyncio
async def test_control_store_rejects_invalid_adapter_id_and_snapshot_provider_id(tmp_path):
    control = ControlStore(tmp_path / "system.db")
    await control.initialize(ProviderConfig(id="seed_provider"))
    provider = await control.get_provider("seed_provider")
    assert provider is not None
    library = await control.create_library(
        {"id": "linked_library", "name": "测试库"}, provider
    )

    with pytest.raises(ValueError, match=r"适配器标识ID.*\[a-zA-Z0-9_-\]"):
        await control.register_adapter_connection(
            library.id,
            adapter_id="Astr bot",
            instance_id="instance-a",
        )

    with pytest.raises(ValueError, match=r"Provider ID.*\[a-zA-Z0-9_-\]"):
        await control.restore_provider_snapshot(
            {
                "providers": [{"id": "provider.bad", "latest_revision": 1}],
                "provider_revisions": [],
            }
        )
