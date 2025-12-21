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
        "bytes": desc.bytes,
        "pages": desc.pages,
        "encrypted": desc.encrypted,
        "metadata": desc.metadata,
        "text_preview": desc.text_preview,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    typer.echo(f"Path: {payload['path']}")
    typer.echo(f"Bytes: {payload['bytes']}")
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
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Compress a PDF (lossless)."""
    if output.exists() and not overwrite:
        raise typer.BadParameter(f"Output already exists: {output} (use --overwrite)")

    result = compress_pdf(pdf, output, linearize=linearize)
    delta = result.output_bytes - result.input_bytes
    typer.echo(f"Wrote: {result.output_path}")
    typer.echo(f"Bytes: {result.input_bytes} -> {result.output_bytes} ({delta:+d})")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

