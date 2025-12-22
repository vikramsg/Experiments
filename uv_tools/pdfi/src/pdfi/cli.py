from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .pdf_ops import compress_pdf, describe_pdf

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def describe(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    max_preview_chars: Annotated[int, typer.Option("--max-preview-chars")] = 800,
) -> None:
    """Describe a PDF (metadata, page count, size, optional text preview)."""
    desc = describe_pdf(pdf, max_preview_chars=max_preview_chars)
    payload = {
        "path": desc.path,
        "kilo_bytes": desc.kilo_bytes,
        "pages": desc.pages,
        "encrypted": desc.encrypted,
        "metadata": desc.metadata,
        "text_preview": desc.text_preview,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    typer.echo(f"Path: {payload['path']}")
    typer.echo(f"Size: {payload['kilo_bytes']:.1f} KB")
    typer.echo(f"Pages: {payload['pages']}")
    typer.echo(f"Encrypted: {payload['encrypted']}")
    if payload["metadata"]:
        typer.echo("Metadata:")
        for k, v in sorted(payload["metadata"].items()):
            typer.echo(f"  {k}: {v}")
    if payload["text_preview"]:
        typer.echo("Text preview:")
        typer.echo(payload["text_preview"])


@app.command()
def compress(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("-o", "--output", dir_okay=False)] = Path("compressed.pdf"),
    linearize: Annotated[bool, typer.Option("--linearize/--no-linearize")] = False,
    aggressive: Annotated[bool, typer.Option("--aggressive/--no-aggressive")] = False,
    dpi: Annotated[int | None, typer.Option("--dpi")] = None,
    jpeg_quality: Annotated[int | None, typer.Option("--jpeg-quality")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """
    Compress a PDF.
    
    By default, uses lossless compression. Use --aggressive for lossy compression
    that renders pages as images (useful for scanned documents to get smaller file sizes).
    """
    if output.exists() and not overwrite:
        raise typer.BadParameter(f"Output already exists: {output} (use --overwrite)")

    if aggressive and (dpi is None or jpeg_quality is None):
        if dpi is None:
            dpi = 120
        if jpeg_quality is None:
            jpeg_quality = 65
        typer.echo(f"Using aggressive compression: DPI={dpi}, JPEG quality={jpeg_quality}")

    result = compress_pdf(
        pdf,
        output,
        linearize=linearize,
        aggressive=aggressive,
        dpi=dpi,
        jpeg_quality=jpeg_quality,
    )
    input_kb = result.input_bytes / 1024
    output_kb = result.output_bytes / 1024
    delta_kb = output_kb - input_kb
    reduction_pct = (delta_kb / input_kb) * 100 if input_kb > 0 else 0
    typer.echo(f"Wrote: {result.output_path}")
    typer.echo(f"Size: {input_kb:.1f} KB -> {output_kb:.1f} KB ({delta_kb:+.1f} KB, {reduction_pct:+.1f}%)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

