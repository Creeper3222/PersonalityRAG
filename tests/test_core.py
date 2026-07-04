from __future__ import annotations

import math

import pytest

import personalityrag.app as app_module
import personalityrag.restart_helper as restart_helper
from personalityrag.app import _normalize_memory_update_payload
from personalityrag.auth import AuthManager, hash_password, verify_password
from personalityrag.config import build_access_url, normalize_access_base_url
from personalityrag.graph import GraphBuilder
from personalityrag.retrieval import rrf_fuse
from personalityrag.schemas import MemoryUpdate, SettingsUpdate


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


def test_settings_payload_keeps_runtime_and_saved_ports_separate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(app_module.config, "access_base_url", "http://127.0.0.1")
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
