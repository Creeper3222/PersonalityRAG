from __future__ import annotations

import math

from personalityrag.auth import AuthManager, hash_password, verify_password
from personalityrag.graph import GraphBuilder
from personalityrag.retrieval import rrf_fuse


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


def test_rrf_is_livingmemory_compatible():
    result = rrf_fuse([(1, 0.9), (2, 0.8)], [(2, 0.95), (3, 0.7)], 60)
    assert result[0][0] == 2
    assert math.isclose(result[0][1], 1 / 62 + 1 / 61)
    assert {item[0] for item in result} == {1, 2, 3}


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
