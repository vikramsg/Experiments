# Training Guide

## Requirements

### Environment
- Python 3.11 or 3.12.
- `uv` for environment management.
- A Hugging Face token (`HF_TOKEN`) if the model repo is gated.

### Dependencies
- `torch`
- `transformers`
- `peft`
- `datasets`
- `librosa`
- `soundfile`
- `evaluate`
- `jiwer`

### Data
- A small labeled ASR dataset (dummy or domain-specific).
- Optional synthetic audio fallback (tone generator) for sanity checks.

## Training Outputs

- Adapter checkpoint (LoRA weights).
- Processor snapshot (tokenizer + feature extractor).
- Metrics report (loss, runtime, device, sample outputs).

## Artifacts Created

- **Adapter checkpoint**: PEFT LoRA weights (directory of safetensors/config).
- **Processor snapshot**: tokenizer + feature extractor.
- **Metrics JSON**: train/eval loss, device, runtime, sample transcripts.

## Final Report Template

```text
# Training Report

## Run Summary
- Date:
- Command:
- Model:
- Dataset:
- Samples:
- Device:

## Training Configuration
- Max steps:
- Batch size:
- Gradient accumulation:
- Learning rate:
- LoRA config:

## Results
- Train loss:
- Eval loss:
- Baseline text (sample):
- Tuned text (sample):
- Elapsed seconds:

## Artifacts
- Adapter path:
- Processor path:
- Metrics path:

## Notes
- Any warnings:
- Observations:
```
