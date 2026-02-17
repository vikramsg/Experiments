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
