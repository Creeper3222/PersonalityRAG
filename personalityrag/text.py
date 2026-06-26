from __future__ import annotations

import re
from pathlib import Path

import jieba


DEFAULT_STOPWORDS = {
    "的",
    "了",
    "和",
    "是",
    "在",
    "我",
    "你",
    "他",
    "她",
    "它",
    "这",
    "那",
    "有",
    "就",
    "也",
    "都",
    "与",
    "及",
    "或",
    "a",
    "an",
    "the",
    "and",
    "or",
    "is",
    "are",
}


class TextProcessor:
    def __init__(self, stopwords_dir: Path | None = None):
        self.stopwords = set(DEFAULT_STOPWORDS)
        if stopwords_dir and stopwords_dir.exists():
            for path in stopwords_dir.glob("*.txt"):
                try:
                    self.stopwords.update(
                        line.strip()
                        for line in path.read_text(
                            encoding="utf-8", errors="ignore"
                        ).splitlines()
                        if line.strip() and not line.startswith("#")
                    )
                except OSError:
                    continue

    def tokenize(self, text: str, remove_stopwords: bool = True) -> list[str]:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        tokens = [
            item.strip()
            for item in jieba.lcut(normalized, cut_all=False)
            if item.strip()
        ]
        if remove_stopwords:
            tokens = [
                token
                for token in tokens
                if token not in self.stopwords and not token.isspace()
            ]
        return tokens

