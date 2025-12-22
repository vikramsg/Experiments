from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import io

import fitz  # PyMuPDF
from PIL import Image


@dataclass(frozen=True)
class PdfDescription:
    path: str
    kilo_bytes: float
    pages: int | None
    encrypted: bool
    metadata: dict[str, str]
    text_preview: str | None


def _normalize_metadata(raw: dict[str, Any] | None) -> dict[str, str]:
    if not raw:
        return {}
    metadata: dict[str, str] = {}
    for key, value in raw.items():
        key_str = str(key).lstrip("/")
        if value is None:
            continue
        metadata[key_str] = str(value)
    return metadata


def describe_pdf(path: str | Path, *, max_preview_chars: int = 800) -> PdfDescription:
    pdf_path = Path(path)
    size_kilo_bytes = pdf_path.stat().st_size / 1024

    doc = fitz.open(str(pdf_path))
    encrypted = doc.is_encrypted

    pages: int | None
    text_preview: str | None

    if encrypted:
        # Try to decrypt with empty password
        if doc.authenticate(""):
            encrypted = False
        else:
            # Can't decrypt, return limited info
            doc.close()
            return PdfDescription(
                path=str(pdf_path),
                kilo_bytes=size_kilo_bytes,
                pages=None,
                encrypted=True,
                metadata={},
                text_preview=None,
            )

    pages = doc.page_count
    text_preview = _extract_preview(doc, max_preview_chars=max_preview_chars)

    metadata = _normalize_metadata(doc.metadata)

    doc.close()
    return PdfDescription(
        path=str(pdf_path),
        kilo_bytes=size_kilo_bytes,
        pages=pages,
        encrypted=encrypted,
        metadata=metadata,
        text_preview=text_preview,
    )


def _extract_preview(doc: fitz.Document, *, max_preview_chars: int) -> str | None:
    if max_preview_chars <= 0:
        return None

    parts: list[str] = []
    remaining = max_preview_chars
    for page_num in range(doc.page_count):
        if remaining <= 0:
            break
        page = doc[page_num]
        text = page.get_text() or ""
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


def _compress_pdf_aggressive(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int,
    jpeg_quality: int,
) -> None:
    """
    Compress PDF using aggressive lossy compression by rendering pages as images.

    Args:
        input_path: Path to input PDF file
        output_path: Path to output PDF file
        dpi: DPI for rendering pages
        jpeg_quality: JPEG quality for image compression (1-100)
    """
    doc = fitz.open(str(input_path))
    new_doc = fitz.open()

    for page in doc:
        # Render page at specified DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("jpeg", jpg_quality=jpeg_quality)

        # Further compress with PIL
        img = Image.open(io.BytesIO(img_data))
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
        compressed_img = output.getvalue()

        # Create new page with compressed image
        new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=compressed_img)

    new_doc.save(str(output_path), deflate=True, garbage=4)
    new_doc.close()
    doc.close()


def compress_pdf(
    input_path: str | Path,
    output_path: str | Path,
    *,
    linearize: bool = False,
    aggressive: bool = False,
    dpi: int | None = None,
    jpeg_quality: int | None = None,
) -> PdfCompressionResult:
    """
    Compress a PDF file.

    Args:
        input_path: Path to input PDF file
        output_path: Path to output PDF file
        linearize: Whether to linearize the PDF for web viewing (not supported, ignored)
        aggressive: Use aggressive compression by rendering pages as images (lossy)
        dpi: DPI for aggressive compression (default: 120 for aggressive mode)
        jpeg_quality: JPEG quality for aggressive compression (default: 65 for aggressive mode)

    Returns:
        PdfCompressionResult with compression statistics
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    input_bytes = in_path.stat().st_size

    if aggressive:
        if dpi is None:
            dpi = 120
        if jpeg_quality is None:
            jpeg_quality = 65

        _compress_pdf_aggressive(in_path, out_path, dpi=dpi, jpeg_quality=jpeg_quality)
    else:
        # Use standard lossless compression with pymupdf
        # Note: linearization is not supported in pymupdf, so linearize parameter is ignored
        doc = fitz.open(str(in_path))
        doc.save(
            str(out_path),
            deflate=True,
            garbage=4,
        )
        doc.close()

    output_bytes = out_path.stat().st_size
    return PdfCompressionResult(
        input_path=str(in_path),
        output_path=str(out_path),
        input_bytes=input_bytes,
        output_bytes=output_bytes,
    )

