from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".bat",
    ".css",
    ".editorconfig",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".lock",
    ".md",
    ".ps1",
    ".py",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
RUSSIAN_TECHNICAL_ONLY_KEYS = {
    "apiBaseUrl",
    "apiKey",
    "loginModeApiKey",
    "providerKindEmbedding",
    "providerKindRerank",
}


def _tracked_text_files() -> list[Path]:
    if not (REPO_ROOT / ".git").exists():
        return [
            path
            for path in REPO_ROOT.rglob("*")
            if path.is_file()
            and not any(
                part in {"__pycache__", ".pytest_cache", ".test-runtime"}
                for part in path.relative_to(REPO_ROOT).parts
            )
            and (
                path.suffix.lower() in TEXT_SUFFIXES
                or path.name == ".editorconfig"
            )
        ]
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    paths = result.stdout.decode("utf-8").split("\0")
    return [
        REPO_ROOT / item
        for item in paths
        if item
        and (
            Path(item).suffix.lower() in TEXT_SUFFIXES
            or Path(item).name == ".editorconfig"
        )
    ]


def _locale(path: Path) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    start = source.index("{")
    end = source.rindex("};") + 1

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate locale key {key!r} in {path.name}")
            result[key] = value
        return result

    parsed = json.loads(source[start:end], object_pairs_hook=reject_duplicates)
    assert all(isinstance(key, str) and isinstance(value, str) for key, value in parsed.items())
    return parsed


def test_all_tracked_text_is_strict_utf8_without_replacement_characters() -> None:
    for path in _tracked_text_files():
        text = path.read_bytes().decode("utf-8", errors="strict")
        assert "\ufffd" not in text, f"replacement character in {path}"


def test_locales_have_identical_unique_key_sets() -> None:
    locales = {
        name: _locale(REPO_ROOT / "static" / "locales" / f"{name}.js")
        for name in ("zh", "en", "ru")
    }
    assert set(locales["en"]) == set(locales["zh"])
    assert set(locales["ru"]) == set(locales["zh"])
    for name, locale in locales.items():
        assert all("\ufffd" not in value for value in locale.values()), name


def test_russian_locale_has_no_cjk_mojibake_or_copied_ui_sentences() -> None:
    russian = _locale(REPO_ROOT / "static" / "locales" / "ru.js")
    for key, value in russian.items():
        assert not re.search(r"[\u3400-\u9fff]", value), f"CJK text in ru.{key}"
        latin_letters = len(re.findall(r"[A-Za-z]", value))
        cyrillic_letters = len(re.findall(r"[\u0400-\u04ff]", value))
        if latin_letters >= 4 and cyrillic_letters == 0:
            assert key in RUSSIAN_TECHNICAL_ONLY_KEYS, (
                f"untranslated Russian locale entry: {key}={value!r}"
            )


def test_html_i18n_references_exist_in_every_locale() -> None:
    html = (REPO_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    referenced = set(
        re.findall(r'data-i18n(?:-placeholder)?="([^"]+)"', html)
    )
    locale_keys = set(_locale(REPO_ROOT / "static" / "locales" / "zh.js"))
    assert referenced <= locale_keys


def test_app_uses_locale_modules_without_runtime_key_overrides() -> None:
    source = (REPO_ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert 'from "./locales/zh.js"' in source
    assert 'from "./locales/en.js"' in source
    assert 'from "./locales/ru.js"' in source
    assert "Object.assign(strings." not in source
