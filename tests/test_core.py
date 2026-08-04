from __future__ import annotations

import math
import os

import pytest
from fastapi import HTTPException

import personalityrag.app as app_module
import personalityrag.indexes as indexes_module
import personalityrag.resource_limits as resource_limits
import personalityrag.restart_helper as restart_helper
from personalityrag.app import _normalize_memory_update_payload
from personalityrag.auth import AuthManager, hash_password, verify_password
from personalityrag.config import (
    AppConfig,
    build_access_url,
    build_adapter_connection_url,
    load_config,
    normalize_access_base_url,
    normalize_bind_host,
    normalize_public_adapter_url,
)
from personalityrag.graph import GraphBuilder
from personalityrag.retrieval import rrf_fuse
from personalityrag.schemas import MemoryUpdate, SettingsUpdate
from personalityrag.service import _graph_k_core_node_ids


def test_auth_session_and_key():
    auth = AuthManager("secret-key", "session-secret", 60)
    assert auth.verify_api_key("secret-key")
    assert not auth.verify_api_key("wrong")
    token = auth.issue_session()
    assert auth.verify_session(token)
    assert not auth.verify_session(token + "x")


def test_webui_password_hash_switches_login_secret():
    encoded = hash_password("correct horse")
    assert "correct horse" not in encoded
    assert verify_password("correct horse", encoded)
    assert not verify_password("wrong horse", encoded)

    auth = AuthManager("api-token", "session-secret", password_hash=encoded)
    assert auth.verify_api_key("api-token")
    assert not auth.verify_login_secret("api-token")
    assert auth.verify_login_secret("correct horse")

    token_only = AuthManager("api-token", "session-secret")
    assert token_only.verify_login_secret("api-token")


def test_access_base_url_normalizes_and_builds_ported_urls():
    assert (
        SettingsUpdate(access_base_url="https://memory.example.com/").access_base_url
        == "https://memory.example.com"
    )
    assert SettingsUpdate(access_base_url="").access_base_url == "http://127.0.0.1"
    assert normalize_access_base_url(" http://127.0.0.1/ ") == "http://127.0.0.1"
    assert (
        build_access_url("https://memory.example.com/", 8766)
        == "https://memory.example.com:8766/"
    )
    with pytest.raises(ValueError):
        normalize_access_base_url("http://127.0.0.1:8765")


def test_public_adapter_url_is_https_origin_and_overrides_local_listener_url():
    assert (
        normalize_public_adapter_url("https://memory.example.com/")
        == "https://memory.example.com"
    )
    assert (
        normalize_public_adapter_url("https://memory.example.com:8443")
        == "https://memory.example.com:8443"
    )
    assert normalize_public_adapter_url("") == ""
    assert (
        build_adapter_connection_url(
            AppConfig(
                access_base_url="http://127.0.0.1",
                public_adapter_url="https://memory.example.com",
                access_port=8766,
            )
        )
        == "https://memory.example.com"
    )
    assert (
        build_adapter_connection_url(
            AppConfig(access_base_url="http://127.0.0.1", access_port=8766)
        )
        == "http://127.0.0.1:8766"
    )
    for invalid in (
        "http://memory.example.com",
        "https://user:pass@memory.example.com",
        "https://memory.example.com/prefix",
        "https://memory.example.com?token=secret",
        "https://memory.example.com#fragment",
    ):
        with pytest.raises(ValueError):
            normalize_public_adapter_url(invalid)


def test_core_listener_host_is_loopback_only():
    assert normalize_bind_host("127.0.0.1") == "127.0.0.1"
    assert normalize_bind_host("LOCALHOST") == "localhost"
    assert normalize_bind_host("::1") == "::1"
    with pytest.raises(ValueError, match="loopback"):
        normalize_bind_host("0.0.0.0")
    with pytest.raises(ValueError, match="loopback"):
        normalize_bind_host("192.168.1.10")


def test_brand_new_config_does_not_persist_a_bootstrap_provider(tmp_path):
    config_path = tmp_path / "config" / "config.json"

    config = load_config(config_path)
    persisted = config_path.read_text(encoding="utf-8")

    assert config.bootstrap_provider_enabled is False
    assert '"provider"' not in persisted
    assert "bge-m3" not in persisted

    reloaded = load_config(config_path)
    assert reloaded.bootstrap_provider_enabled is False


def test_faiss_thread_limit_uses_adaptive_default_and_accepts_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PERSONALITYRAG_FAISS_THREADS", raising=False)
    assert indexes_module._configured_faiss_threads() == min(
        4, resource_limits.effective_cpu_count()
    )
    monkeypatch.setenv("PERSONALITYRAG_FAISS_THREADS", "3")
    assert indexes_module._configured_faiss_threads() == 3
    monkeypatch.setenv("PERSONALITYRAG_FAISS_THREADS", "invalid")
    assert indexes_module._configured_faiss_threads() == min(
        4, resource_limits.effective_cpu_count()
    )


def test_io_and_blas_limits_accept_environment_overrides(monkeypatch):
    monkeypatch.setenv("PERSONALITYRAG_IO_WORKERS", "3")
    monkeypatch.setenv("PERSONALITYRAG_BLAS_THREADS", "2")
    assert resource_limits.configured_io_workers() == 3
    assert resource_limits.configured_blas_threads() == 2
    assert resource_limits.configure_numeric_thread_environment() == 2
    assert os.environ["OPENBLAS_NUM_THREADS"] == "2"


def test_settings_payload_keeps_runtime_and_saved_ports_separate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(app_module.config, "access_base_url", "http://127.0.0.1")
    monkeypatch.setattr(app_module.config, "public_adapter_url", "")
    monkeypatch.setattr(app_module.config, "port", 8765)
    monkeypatch.setattr(app_module.config, "access_port", 8767)
    monkeypatch.setenv("PERSONALITYRAG_ACTUAL_PORT", "8765")
    monkeypatch.setenv("PERSONALITYRAG_ACCESS_ACTUAL_PORT", "8766")

    payload = app_module._settings_payload()

    assert payload["webui_url"] == "http://127.0.0.1:8765/"
    assert payload["configured_webui_url"] == "http://127.0.0.1:8765/"
    assert payload["api_access_url"] == "http://127.0.0.1:8766/"
    assert payload["configured_api_access_url"] == "http://127.0.0.1:8767/"
    assert payload["actual_access_port"] == 8766
    assert payload["configured_access_port"] == 8767
    assert payload["public_adapter_url"] == ""
    assert payload["recommended_adapter_url"] == "http://127.0.0.1:8766"


def test_restart_probe_urls_follow_configured_webui_port(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(app_module.config, "access_base_url", "http://127.0.0.1")
    monkeypatch.setattr(app_module.config, "port", 8765)

    urls = app_module._restart_probe_urls()

    assert urls[0] == "http://127.0.0.1:8765/"
    assert urls[1] == "http://127.0.0.1:8766/"
    assert len(urls) == app_module.RESTART_PROBE_SCAN_LIMIT + 1


def test_restart_helper_prefers_launcher_on_windows_repo_layout(tmp_path):
    root = tmp_path / "PersonalityRAG"
    scripts = root / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    launcher = root / "launcher.bat"
    launcher.write_text("@echo off\r\n", encoding="utf-8")
    (scripts / "python.exe").write_text("", encoding="utf-8")

    command = restart_helper._restart_service_command(root)

    assert command == ["cmd.exe", "/c", str(launcher)]


def test_rrf_is_livingmemory_compatible():
    result = rrf_fuse([(1, 0.9), (2, 0.8)], [(2, 0.95), (3, 0.7)], 60)
    assert result[0][0] == 2
    assert math.isclose(result[0][1], 1 / 62 + 1 / 61)
    assert {item[0] for item in result} == {1, 2, 3}


def test_memory_update_importance_scale_compatibility():
    assert _normalize_memory_update_payload(
        MemoryUpdate(importance=1.0)
    )["importance"] == 1.0
    assert _normalize_memory_update_payload(
        MemoryUpdate(importance=10.0, value_scale="display")
    )["importance"] == 1.0
    assert _normalize_memory_update_payload(
        MemoryUpdate(importance=1.0, value_scale="display")
    )["importance"] == 0.1
    assert _normalize_memory_update_payload(
        MemoryUpdate(importance=0.25, value_scale="stored")
    )["importance"] == 0.25


def test_generic_memory_update_rejects_persona_field():
    with pytest.raises(HTTPException) as exc_info:
        _normalize_memory_update_payload(MemoryUpdate(persona_id="贝雷特"))

    assert exc_info.value.status_code == 400
    assert "独立的人格编辑接口" in str(exc_info.value.detail)


def test_graph_builder_legacy_shape():
    graph = GraphBuilder().build(
        7,
        "澄月喜欢星空。",
        {
            "canonical_summary": "澄月喜欢星空。",
            "topics": ["星空"],
            "participants": ["澄月"],
            "key_facts": ["澄月喜欢星空"],
            "persona_id": "贝雷特",
            "session_id": "test:GroupMessage:1",
            "importance": 0.8,
        },
    )
    assert {node["node_type"] for node in graph["nodes"]} == {
        "topic",
        "person",
        "fact",
    }
    assert {edge["relation_type"] for edge in graph["edges"]} == {
        "describes",
        "mentioned_in",
    }
    assert all(entry["source_memory_id"] == 7 for entry in graph["entries"])


def test_graph_builder_uses_stable_accounts_and_suppresses_alias_topics():
    builder = GraphBuilder()
    metadata = {
        "canonical_summary": "Alice approved the release plan.",
        "topics": ["Alice", "release"],
        "participants": ["Alice"],
        "participant_identities": [
            {
                "platform": "OneBot",
                "sender_id": "42",
                "display_name": "Alice",
                "aliases": ["Alicia"],
                "is_bot": False,
            }
        ],
        "key_facts": ["Alice approved the release plan."],
    }

    first = builder.build(8, metadata["canonical_summary"], metadata)
    second = builder.build(
        9,
        metadata["canonical_summary"],
        {
            **metadata,
            "participant_identities": [
                {
                    **metadata["participant_identities"][0],
                    "display_name": "Alicia",
                    "aliases": ["Alice"],
                }
            ],
        },
    )

    first_person = next(node for node in first["nodes"] if node["node_type"] == "person")
    second_person = next(node for node in second["nodes"] if node["node_type"] == "person")
    assert first_person["node_key"] == second_person["node_key"] == "person:account:onebot:42"
    assert first_person["metadata"] == {
        "identity_key": "onebot:42",
        "sender_id": "42",
        "platform": "onebot",
        "aliases": ["Alicia", "Alice"],
        "is_bot": False,
    }
    assert {node["value"] for node in first["nodes"] if node["node_type"] == "topic"} == {"release"}
    assert any(edge["relation_type"] == "mentioned_in" for edge in first["edges"])


def test_graph_page_k_core_removes_isolated_and_single_link_nodes():
    edges = [
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},
        {"source": 3, "target": 1},
        {"source": 3, "target": 4},
    ]

    assert _graph_k_core_node_ids({1, 2, 3, 4, 5}, edges, 2) == {1, 2, 3}
