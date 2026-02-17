## Moonshine LoRA POC

Quick proof-of-concept runner that fine-tunes a Moonshine checkpoint on the
LibriSpeech dummy dataset (or a synthetic fallback) and saves a LoRA adapter.

### Setup

- `uv venv`
- `source .venv/bin/activate`
- `uv sync`

### Run

- `HF_TOKEN=... uv run python main.py` (required if the Moonshine repo is gated)

Artifacts are written to `outputs/poc` by default.

## Documentation

### POC Flow

1. Creates a tiny LibriSpeech dummy dataset (or synthetic fallback).
2. Loads Moonshine + LoRA adapters.
3. Trains for a short run.
4. Runs before/after inference.
5. Saves adapter + processor + metrics.

### Verification

- Run: `HF_TOKEN=... uv run python main.py --dataset librispeech_dummy --dataset-samples 8`
- Verify training happened by checking:
  - `outputs/poc/poc_metrics.json` contains `train_loss`, `eval_loss`, `elapsed_seconds`.
  - `outputs/poc/lora_adapter` and `outputs/poc/processor` exist.

### Results (Example)

- `train_loss`: ~7.99
- `eval_loss`: ~6.12
- `device`: `mps`
- `elapsed_seconds`: ~23.46

### Outputs

- `outputs/poc/lora_adapter`
- `outputs/poc/processor`
- `outputs/poc/poc_metrics.json`
