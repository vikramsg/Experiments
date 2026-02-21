# Training Guide

## Purpose & Scope

- Fine-tune Moonshine ASR models with LoRA adapters for domain adaptation.
- Optimize for domain WER/CER improvements while staying within Apple Silicon memory limits.

## Glossary

- Domain manifest: The primary evaluation dataset that represents domain-shifted production-like audio (for this repo: `data/domain_manifest.jsonl`).
- Heldout manifest: The safety/guardrail evaluation dataset used to detect regressions on non-target data (for this repo: `data/heldout_manifest.jsonl`).
- Baseline: Metrics from the base model before applying LoRA training.
- Tuned model: The same base model with LoRA adapter weights applied after training.
- Guardrail metric: A metric tracked to prevent regressions while optimizing another target metric.
- Headroom: The remaining achievable improvement on a dataset; low headroom usually means little expected WER gain.
- Normalization/decode parity: Using the same audio normalization and decode settings across training, evaluation, and inference to keep comparisons valid.
- Manifest: A JSONL file describing evaluation/training samples, typically with audio path and transcript metadata.

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
- Ensure training preprocessing matches inference preprocessing (e.g., RMS normalization) to avoid distribution drift.
- Keep LoRA runs long enough to move WER (small runs can regress even on in-domain data).
- Consider modern LoRA variants when stability or generalization is a concern:
  - **DoRA** (`use_dora=True` in `LoraConfig`) for better stability and reduced hyperparameter sensitivity.
  - **PiSSA** (`init_lora_weights="pissa"`) to initialize adapters with principal singular values.
  - **Flat-LoRA** style noise/perturbation (if supported by the training stack) to reduce sharp minima.

## Evaluation Plan

- Metrics: WER (primary), CER (secondary), loss curves.
- Baseline: evaluate base model on validation set before training.
- Success measured against explicit thresholds (see report template).
- Keep decode settings consistent across baseline, tuned, and in-training evals (beam size, penalties, max tokens).
- When using Moonshine, match inference normalization in evaluation runs.

## Outputs & Artifacts

- Adapter checkpoint (LoRA weights).
- Processor snapshot (tokenizer + feature extractor).
- Metrics report (loss, runtime, device, sample outputs).
- Optional: merged model checkpoint for deployment.
- Capture the exact manifest used for baseline/tuned STT to ensure reproducibility.

## Debugging WER Regression

- Re-run baseline and tuned STT on the same manifest to confirm the regression persists.
- Compare per-sample WER deltas to see if the tuned adapter drifts on in-domain data.
- Check for preprocessing mismatch: training data loader currently uses raw audio, while inference applies RMS normalization.
- Validate that the evaluation path uses the same Moonshine decode settings used in `src/lora_training/transcribe.py`.
- Increase training steps/samples before concluding a regression is model-related.
- If regressions persist, try DoRA or PiSSA initialization to improve convergence stability.

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
- Training duration (h:m:s):
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
