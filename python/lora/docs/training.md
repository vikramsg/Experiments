# Training Guide

## Purpose & Scope

- Fine-tune Moonshine ASR models with LoRA adapters for domain adaptation.
- Optimize for domain WER/CER improvements while staying within Apple Silicon memory limits.

## Data Requirements

- Labeled ASR dataset: audio files + transcripts.
- Train/validation/test split (e.g., 80/10/10).
- Audio: 16 kHz mono WAV/FLAC recommended; normalize loudness.
- Keep a small “quick check” subset (20–100 samples) for POC runs.

## Training Configuration

- Model: Moonshine tiny/base/medium (start tiny for POC).
- LoRA: start with attention projections (`q_proj`, `v_proj`).
- Batch size: 1–2 on MPS; use gradient accumulation.
- Precision: fp16 on MPS where stable.

## Evaluation Plan

- Metrics: WER (primary), CER (secondary), loss curves.
- Baseline: evaluate base model on validation set before training.
- Success measured against explicit thresholds (see report template).

## Outputs & Artifacts

- Adapter checkpoint (LoRA weights).
- Processor snapshot (tokenizer + feature extractor).
- Metrics report (loss, runtime, device, sample outputs).
- Optional: merged model checkpoint for deployment.

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

## Hypothesis
- What improvement is expected and why:

## Success Criteria
- Explicit thresholds to accept/reject hypothesis:

## Training Configuration
- Max steps:
- Batch size:
- Gradient accumulation:
- Learning rate:
- LoRA config:

## Evaluation Plan
- Baseline metrics:
- Post-training metrics:
- Comparison method:

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

## Decision
- Meets success criteria? (yes/no)
- Next steps:

## Notes
- Any warnings:
- Observations:
```
