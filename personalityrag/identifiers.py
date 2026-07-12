from __future__ import annotations

import re
from typing import Any


IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
IDENTIFIER_RULE_MESSAGE = "只能使用[a-zA-Z0-9_-]中的字符来命名ID"


def validate_identifier(value: Any, *, field: str = "ID") -> str:
    """Return an identifier only when it is safe for URLs, paths, and keys."""
    text = value if isinstance(value, str) else "" if value is None else str(value)
    if not IDENTIFIER_PATTERN.fullmatch(text):
        raise ValueError(f"{field}{IDENTIFIER_RULE_MESSAGE}")
    return text


def is_valid_identifier(value: Any) -> bool:
    try:
        validate_identifier(value)
    except ValueError:
        return False
    return True
