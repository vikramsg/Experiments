# Experiments

Personal experiments and notes across multiple languages and tools.

## Image editor

Web image editor (multi-layer compositor + single-image crop) lives in `client_tools/image-editor/`.

Live site: https://vikramsg.github.io/Experiments/

## uv tools

The `uv_tools/` directory contains small Python CLIs intended to be installed via `uv tool install`.

### pdfi

PDF utility CLI (describe and compress PDFs).

Install directly from this repo (monorepo subdirectory):

```bash
uv tool install "pdfi @ git+https://github.com/vikramsg/Experiments.git@master#subdirectory=uv_tools/pdfi"
```

See `uv_tools/pdfi/README.md` for usage and development.
