from __future__ import annotations

import json
from pathlib import Path

import pytest
from reportlab.pdfgen.canvas import Canvas
from typer.testing import CliRunner

from pdfi.cli import app
from pdfi.pdf_ops import compress_pdf, describe_pdf


@pytest.fixture()
def make_uncompressed_pdf(tmp_path: Path):
    def _make(*, name: str, text: str) -> Path:
        path = tmp_path / name
        canvas = Canvas(str(path), pageCompression=0)
        canvas.setFont("Helvetica", 12)
        canvas.drawString(72, 720, text)
        canvas.save()
        return path

    return _make


def test_describe_pdf_extracts_metadata_and_preview(make_uncompressed_pdf) -> None:
    pdf_path = make_uncompressed_pdf(name="sample.pdf", text="Hello PDF tool")

    desc = describe_pdf(pdf_path, max_preview_chars=200)
    assert desc.bytes > 0
    assert desc.pages == 1
    assert desc.encrypted is False
    assert desc.text_preview is not None
    assert "Hello" in desc.text_preview


def test_compress_pdf_reduces_size_for_uncompressed_streams(tmp_path: Path, make_uncompressed_pdf) -> None:
    # Lots of repeated text, stored without stream compression (pageCompression=0).
    pdf_path = make_uncompressed_pdf(name="big.pdf", text=("A" * 20000))
    out_path = tmp_path / "big.compressed.pdf"

    result = compress_pdf(pdf_path, out_path)
    assert Path(result.output_path).exists()
    assert result.output_bytes < result.input_bytes

    desc_in = describe_pdf(pdf_path, max_preview_chars=0)
    desc_out = describe_pdf(out_path, max_preview_chars=0)
    assert desc_in.pages == desc_out.pages == 1


def test_cli_describe_json(make_uncompressed_pdf) -> None:
    pdf_path = make_uncompressed_pdf(name="cli.pdf", text="CLI JSON")

    runner = CliRunner()
    res = runner.invoke(app, ["describe", str(pdf_path), "--json", "--max-preview-chars", "50"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["pages"] == 1
    assert payload["encrypted"] is False
    assert "CLI" in (payload["text_preview"] or "")


def test_cli_compress(tmp_path: Path, make_uncompressed_pdf) -> None:
    pdf_path = make_uncompressed_pdf(name="cli-big.pdf", text=("B" * 15000))
    out_path = tmp_path / "cli-out.pdf"

    runner = CliRunner()
    res = runner.invoke(app, ["compress", str(pdf_path), "-o", str(out_path)])
    assert res.exit_code == 0, res.output
    assert out_path.exists()
    assert out_path.stat().st_size < pdf_path.stat().st_size
