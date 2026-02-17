## Moonshine LoRA POC

Quick proof-of-concept runner that fine-tunes a Moonshine checkpoint on a tiny
synthetic audio dataset and saves a LoRA adapter.

### Setup

- `uv venv`
- `source .venv/bin/activate`
- `uv sync`

### Run

- `HF_TOKEN=... uv run python main.py` (required if the Moonshine repo is gated)

Artifacts are written to `outputs/poc` by default.
