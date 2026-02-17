## Moonshine LoRA

Fine-tunes a Moonshine checkpoint using LoRA. 

### Setup

Prefer `make` targets when available.

- `make venv`
- `source .venv/bin/activate`
- `make sync`

### Run

Prefer `make` targets when available.

- `HF_TOKEN=... make run` (required if the Moonshine repo is gated)

Small real-world run:

- `HF_TOKEN=... uv run python train_real_small.py`

Artifact STT check:

- `uv run python scripts/build_manifest.py --split test.clean --samples 5 --output data/heldout_manifest.jsonl`
- `uv run python scripts/run_stt.py --model-id UsefulSensors/moonshine-tiny --adapter-dir outputs/real_small/lora_adapter --processor-dir outputs/real_small/processor --audio-list data/heldout_manifest.jsonl --output outputs/real_small/artifact_test.json`

Artifacts are written to `outputs` by default.

## Documentation

### Training Guide

Use `docs/training.md` for full requirements, evaluation plan, and the report
template to capture results for each run.

### Recommended Run Flow

1. Prepare dataset and confirm splits.
2. Evaluate baseline metrics on validation data.
3. Train LoRA adapter and capture metrics.
4. Re-evaluate and compare to success criteria.
5. Record results using the training report template.
