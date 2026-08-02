from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import io
import math
from pathlib import Path
import re
import unicodedata

from markitdown_no_magika import MarkItDown, StreamInfo
from pypdf import PdfReader

from .text import normalize_text


MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
MAX_EXTRACTED_CHARACTERS = 5_000_000
SUPPORTED_DOCUMENT_SUFFIXES = frozenset(
    {".txt", ".md", ".markdown", ".pdf", ".docx"}
)

_URL_ONLY = re.compile(r"^https?://\S+$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedDocument:
    content: str
    format_hint: str
    parser_id: str
    page_count: int | None = None


def _normalized_source_text(value: str) -> str:
    return normalize_text(
        unicodedata.normalize("NFKC", str(value or "")).replace("\x00", "")
    )


def _parse_utf8(data: bytes, *, markdown: bool) -> ParsedDocument:
    try:
        content = _normalized_source_text(data.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise ValueError("TXT / Markdown documents must use UTF-8 encoding") from exc
    return ParsedDocument(
        content=content,
        format_hint="markdown" if markdown else "text",
        parser_id="utf8_source_v1",
    )


def _boundary_lines(lines: list[str]) -> set[str]:
    nonempty = [line.strip() for line in lines if line.strip()]
    return set(nonempty[:3] + nonempty[-3:])


def _parse_pdf(data: bytes) -> ParsedDocument:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"PDF document cannot be opened: {exc}") from exc
    if reader.is_encrypted:
        raise ValueError("Encrypted PDF documents are not supported")

    pages: list[list[str]] = []
    boundary_counts: Counter[str] = Counter()
    for page in reader.pages:
        try:
            extracted = page.extract_text() or ""
        except Exception as exc:
            raise ValueError(f"PDF text extraction failed: {exc}") from exc
        normalized = unicodedata.normalize("NFKC", extracted).replace("\x00", "")
        lines = [line.strip() for line in normalized.splitlines()]
        pages.append(lines)
        boundary_counts.update(_boundary_lines(lines))

    minimum_repetitions = max(3, math.ceil(len(pages) * 0.05))
    repeated_boundary = {
        line
        for line, count in boundary_counts.items()
        if line and count >= minimum_repetitions
    }
    rendered_pages: list[str] = []
    extracted_characters = 0
    for page_number, lines in enumerate(pages, start=1):
        nonempty_indexes = [index for index, line in enumerate(lines) if line]
        boundary_indexes = set(nonempty_indexes[:3] + nonempty_indexes[-3:])
        cleaned: list[str] = []
        for index, line in enumerate(lines):
            if not line:
                if cleaned and cleaned[-1]:
                    cleaned.append("")
                continue
            if _URL_ONLY.fullmatch(line):
                continue
            if index in boundary_indexes and line in repeated_boundary:
                continue
            cleaned.append(line)
        body = normalize_text("\n".join(cleaned))
        if body:
            extracted_characters += sum(1 for char in body if not char.isspace())
            rendered_pages.append(f"〔PDF 第 {page_number} 页〕\n\n{body}")

    content = normalize_text("\n\n".join(rendered_pages))
    if extracted_characters < 80:
        raise ValueError(
            "PDF contains no usable extractable text; scanned PDF OCR is not supported"
        )
    if len(content) > MAX_EXTRACTED_CHARACTERS:
        raise ValueError("Extracted PDF text exceeds 5,000,000 characters")
    return ParsedDocument(
        content=content,
        format_hint="text",
        parser_id="pypdf_text_v1",
        page_count=len(pages),
    )


def _parse_docx(data: bytes, filename: str) -> ParsedDocument:
    try:
        converter = MarkItDown(enable_plugins=False)
        result = converter.convert(
            io.BytesIO(data),
            stream_info=StreamInfo(extension=".docx", filename=filename),
        )
    except Exception as exc:
        raise ValueError(f"DOCX document cannot be converted: {exc}") from exc
    content = _normalized_source_text(result.markdown or "")
    if len(content) > MAX_EXTRACTED_CHARACTERS:
        raise ValueError("Converted DOCX text exceeds 5,000,000 characters")
    return ParsedDocument(
        content=content,
        format_hint="markdown",
        parser_id="markitdown_docx_v1",
    )


def parse_document_bytes(filename: str, data: bytes) -> ParsedDocument:
    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise ValueError(f"Unsupported document format: {safe_name}")
    if not data or len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"Document is empty or exceeds 20 MiB: {safe_name}")
    if suffix == ".txt":
        parsed = _parse_utf8(data, markdown=False)
    elif suffix in {".md", ".markdown"}:
        parsed = _parse_utf8(data, markdown=True)
    elif suffix == ".pdf":
        parsed = _parse_pdf(data)
    else:
        parsed = _parse_docx(data, safe_name)
    if not parsed.content:
        raise ValueError(f"Document has no indexable text: {safe_name}")
    return parsed
