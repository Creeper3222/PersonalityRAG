from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


MIN_VALID_CONTEXT_TOKENS = 128
MANUAL_CONTEXT_FALLBACK_TOKENS = 512
STATIC_CONTEXT_TABLE_PATH = (
    Path(__file__).resolve().parent
    / "resources"
    / "embedding_context_lengths.json"
)


def _normalize_model_name(value: str | None) -> str:
    return str(value or "").strip().casefold()


@lru_cache(maxsize=1)
def _load_static_table() -> dict[str, Any]:
    try:
        raw = STATIC_CONTEXT_TABLE_PATH.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return {
            "version": "missing",
            "sha256": "",
            "models": {},
        }
    models = payload.get("models") if isinstance(payload, dict) else {}
    if not isinstance(models, dict):
        models = {}
    return {
        "version": str(payload.get("version") or "unknown"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "models": {
            str(name): int(tokens)
            for name, tokens in models.items()
            if not isinstance(tokens, bool)
            and str(name or "").strip()
            and int(tokens or 0) >= MIN_VALID_CONTEXT_TOKENS
        },
    }


def static_context_table_metadata() -> dict[str, str]:
    table = _load_static_table()
    return {
        "version": str(table.get("version") or "unknown"),
        "sha256": str(table.get("sha256") or ""),
    }


def lookup_static_context_length(model_name: str | None) -> dict[str, Any]:
    """Look up an embedding context window from the read-only MTEB-derived table.

    Matching is intentionally conservative:
    - exact full model name wins;
    - otherwise a basename match is accepted only when it is unique.
    """
    query = _normalize_model_name(model_name)
    if not query:
        return {"max_context_tokens": 0, "max_context_tokens_source": ""}
    table = _load_static_table()
    models: dict[str, int] = table.get("models") or {}
    exact = {
        _normalize_model_name(name): (name, tokens)
        for name, tokens in models.items()
    }
    if query in exact:
        name, tokens = exact[query]
        return {
            "max_context_tokens": int(tokens),
            "max_context_tokens_source": f"auto:mteb:{table['version']}:{name}",
        }
    basename = query.rsplit("/", 1)[-1]
    matches = [
        (name, tokens)
        for name, tokens in models.items()
        if _normalize_model_name(name).rsplit("/", 1)[-1] == basename
    ]
    if len(matches) == 1:
        name, tokens = matches[0]
        return {
            "max_context_tokens": int(tokens),
            "max_context_tokens_source": f"auto:mteb:{table['version']}:{name}",
        }
    return {"max_context_tokens": 0, "max_context_tokens_source": ""}
