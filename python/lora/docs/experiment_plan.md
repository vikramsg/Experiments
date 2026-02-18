# Experiment Plan

Single source of truth for what we want to test, why, and how we decide
whether it worked. Keep this aligned with `docs/training.md` guidance and
reference concrete run outputs in `docs/training_report.md`.

## Goals
- Improve WER on target-domain audio while keeping baseline performance stable.
- Avoid train/inference mismatch (normalization + decode settings).
- Ensure evaluation calls respect active LoRA adapters.
- Produce reproducible improvements across at least two runs.

## Evaluation Protocol (Do Not Deviate)
1. **Manifest parity**: baseline and tuned WER must use the same manifest.
2. **Normalization parity**: RMS normalization must match inference behavior.
3. **Decode parity**: use the same Moonshine decode settings everywhere.
4. **Adapter parity**: evaluation must call adapter-aware `generate` when LoRA is active.
5. **Coverage**: use full-manifest WER for primary decisions; tiny subsets are POC only.

## Known Risks / Failure Modes
- **PEFT bypass in eval**: `unwrap_peft(model)` uses `base_model.generate`, which can
  skip adapter-aware generation when adapters are active.
- **Clean dataset saturation**: `librispeech_asr` clean manifests have baseline WER ≈ 4–5%,
  leaving little headroom for improvement.
- **Loss ≠ WER**: improving loss can still worsen WER if decoding is misaligned.
- **Small-run noise**: short runs can regress WER even with stable loss curves.

## Decision Criteria
- **Primary**: tuned WER < baseline WER on the main heldout manifest.
- **Secondary**: tuned WER is non-worse on at least one additional manifest.
- **Reproducibility**: same improvement holds for two runs with fixed seeds.
- **Sanity**: training loss decreases without catastrophic WER regression (>2×).

## Hypotheses and Tests

### H1 — Adapter-aware Generation in Evaluation
- **Problem**: `src/lora/evaluation.py` unwraps PEFT and calls `base_model.generate`.
- **Risk**: LoRA adapter configuration may be ignored during evaluation.
- **Change**: call `model.generate` (PeftModel) when adapters are active.
- **Expected**: in-training WER aligns with `scripts/run_stt.py` WER direction.
- **Success**: tuned WER from training eval matches the trend from `run_stt.py`.

### H2 — Training Volume for 1k Samples
- **Problem**: 50–200 updates are unlikely to converge; WER is noisy.
- **Change**: use ~2000 steps (batch 1, grad_accum 4, 1000 samples ≈ 8 epochs).
- **Expected**: clearer convergence/overfitting signal; WER trend stabilizes.
- **Success**: tuned WER improves or overfits in a predictable monotonic trend.

### H3 — Learning Rate Revisit Post-Normalization
- **Problem**: prior regression may have been normalization mismatch, not LR.
- **Change**: test 1e-4 and 3e-4 with normalization + decode fixed.
- **Expected**: faster loss convergence without WER collapse.
- **Success**: tuned WER improves or stays within ±5% of baseline.

### H4 — Harder / Domain-Shifted Evaluation
- **Problem**: clean LibriSpeech is near model capacity; limited headroom.
- **Change**: add a noisy/accents/domain manifest.
- **Expected**: tuned WER improves on domain shift even if clean stays flat.
- **Success**: tuned WER improves on domain manifest while clean stays non-worse.

## Test Matrix (Fill Per Run)
| Run ID | Date | Command | Dataset | Steps | LR | LoRA Targets | Manifest | Baseline WER | Tuned WER | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |

## Run Notes
- Record any non-standard action, fallback, or manual intervention with a date.
