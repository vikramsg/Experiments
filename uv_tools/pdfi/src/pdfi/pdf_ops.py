from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pikepdf
from pypdf import PdfReader


@dataclass(frozen=True)
class PdfDescription:
    path: str
    bytes: int
    pages: int | None
    encrypted: bool
    metadata: dict[str, str]
    text_preview: str | None


def _normalize_metadata(raw: Any) -> dict[str, str]:
    if not raw:
        return {}
    metadata: dict[str, str] = {}
    for key, value in dict(raw).items():
        key_str = str(key).lstrip("/")
        if value is None:
            continue
        metadata[key_str] = str(value)
    return metadata


def describe_pdf(path: str | Path, *, max_preview_chars: int = 800) -> PdfDescription:
    pdf_path = Path(path)
    size_bytes = pdf_path.stat().st_size

    reader = PdfReader(str(pdf_path))
    encrypted = bool(getattr(reader, "is_encrypted", False))

    pages: int | None
    text_preview: str | None

    if encrypted:
        try:
            reader.decrypt("")  # type: ignore[attr-defined]
            encrypted = False
        except Exception:
            pages = None
            text_preview = None
        else:
            pages = len(reader.pages)
            text_preview = _extract_preview(reader, max_preview_chars=max_preview_chars)
    else:
        pages = len(reader.pages)
        text_preview = _extract_preview(reader, max_preview_chars=max_preview_chars)

    metadata = _normalize_metadata(getattr(reader, "metadata", None))

    return PdfDescription(
        path=str(pdf_path),
        bytes=size_bytes,
        pages=pages,
        encrypted=encrypted,
        metadata=metadata,
        text_preview=text_preview,
    )


def _extract_preview(reader: PdfReader, *, max_preview_chars: int) -> str | None:
    if max_preview_chars <= 0:
        return None

    parts: list[str] = []
    remaining = max_preview_chars
    for page in reader.pages:
        if remaining <= 0:
            break
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = " ".join(text.split())
        if not text:
            continue
        if len(text) > remaining:
            parts.append(text[:remaining])
            remaining = 0
            break
        parts.append(text)
        remaining -= len(text)

    if not parts:
        return None
    return " ".join(parts)


@dataclass(frozen=True)
class PdfCompressionResult:
    input_path: str
    output_path: str
    input_bytes: int
    output_bytes: int


def compress_pdf(
    input_path: str | Path,
    output_path: str | Path,
    *,
    linearize: bool = False,
) -> PdfCompressionResult:
    in_path = Path(input_path)
    out_path = Path(output_path)

    input_bytes = in_path.stat().st_size

    with pikepdf.open(str(in_path)) as pdf:
        pdf.save(
            str(out_path),
            linearize=linearize,
            compress_streams=True,
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
        )

    output_bytes = out_path.stat().st_size
    return PdfCompressionResult(
        input_path=str(in_path),
        output_path=str(out_path),
        input_bytes=input_bytes,
        output_bytes=output_bytes,
    )

