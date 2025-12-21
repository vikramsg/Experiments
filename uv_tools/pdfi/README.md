# pdf-describe-compress

`pdf-tool` is a small CLI for:

- Describing PDFs (metadata, page count, size, optional text preview)
- Compressing PDFs (lossless, via qpdf/pikepdf)

## Install (as a uv tool)

```bash
uv tool install pdf-describe-compress
```

## Install from a git repo (as a uv tool)

```bash
uv tool install "pdf-describe-compress @ git+https://github.com/<org>/<repo>.git"
```

### Monorepos / subdirectory installs

If your repo is a monorepo, you can point `uv` at a subdirectory:

```bash
uv tool install "pdf-describe-compress @ git+https://github.com/<org>/<repo>.git@<rev>#subdirectory=path/to/project"
```

The `subdirectory` must be a buildable Python package root with a `pyproject.toml` that includes:

- `[build-system]` (a PEP 517 build backend, e.g. hatchling or setuptools)
- `[project]` (name/version/dependencies)
- A backend config that includes your package code in the wheel (commonly `src/<package_name>/...`)

If you want an executable installed by `uv tool install`, also define `[project.scripts]` (or `[project.gui-scripts]`).

## Local development

```bash
uv run pytest
uv run pdf-tool describe path/to/file.pdf
uv run pdf-tool compress path/to/in.pdf -o path/to/out.pdf
```
