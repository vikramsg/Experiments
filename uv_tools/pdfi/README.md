# pdfi

`pdfi` is a small CLI for:

- Describing PDFs (metadata, page count, size, optional text preview)
- Compressing PDFs (lossless compression via pymupdf)
- Aggressive compression (lossy, renders pages as images for scanned documents)

## Install from a git repo (as a uv tool)

```bash
uv tool install "pdfi @ git+https://github.com/vikramsg/Experiments.git@master#subdirectory=uv_tools/pdfi"
```

### Monorepos / subdirectory installs

If your repo is a monorepo, you can point `uv` at a subdirectory, and pin to a branch, tag, or commit:

```bash
uv tool install "pdfi @ git+https://github.com/vikramsg/Experiments.git@<rev>#subdirectory=uv_tools/pdfi"
```

Note: `uv tool install` expects a PEP 508 direct reference for Git installs, so you must include `git+` and the package name (`pdfi @ ...`).

The `subdirectory` must be a buildable Python package root with a `pyproject.toml` that includes:

- `[build-system]` (a PEP 517 build backend, e.g. hatchling or setuptools)
- `[project]` (name/version/dependencies)
- A backend config that includes your package code in the wheel (commonly `src/<package_name>/...`)

If you want an executable installed by `uv tool install`, also define `[project.scripts]` (or `[project.gui-scripts]`).

## Local development

```bash
uv run pytest
uv run pdfi describe path/to/file.pdf
uv run pdfi compress path/to/in.pdf -o path/to/out.pdf
```

## Usage

### Describe a PDF

```bash
pdfi describe file.pdf
pdfi describe file.pdf --json
pdfi describe file.pdf --max-preview-chars 200
```

### Compress a PDF

**Lossless compression** (default):
```bash
pdfi compress input.pdf -o output.pdf
```

**Aggressive compression** (lossy, for scanned documents):
```bash
# Use default settings (120 DPI, JPEG quality 65)
pdfi compress input.pdf -o output.pdf --aggressive

# Custom DPI and JPEG quality
pdfi compress input.pdf -o output.pdf --aggressive --dpi 100 --jpeg-quality 60
```

Aggressive compression renders each page as an image and compresses it, which can significantly reduce file size for scanned documents or PDFs with many images. This is lossy compression - text becomes unsearchable and quality may be reduced, but file sizes can be reduced by 40-60% or more.

**Options:**
- `--dpi`: DPI for rendering pages (default: 120 when using `--aggressive`)
- `--jpeg-quality`: JPEG quality 1-100 (default: 65 when using `--aggressive`)
- `--linearize`: Linearize PDF for web viewing (currently not supported, ignored)
- `--overwrite`: Overwrite output file if it exists
