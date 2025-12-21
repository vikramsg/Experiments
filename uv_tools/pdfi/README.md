# pdfi

`pdfi` is a small CLI for:

- Describing PDFs (metadata, page count, size, optional text preview)
- Compressing PDFs (lossless, via qpdf/pikepdf)

## Install (as a uv tool)

```bash
uv tool install pdfi
```

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
