# Plan: LoRA Tuning Experiments (Hypothesis-Driven)

## Goal
Establish a reproducible tuning workflow that improves WER on target-domain data
while keeping baseline performance stable, using a unified experiment runner
instead of shared/legacy scripts.

## Reference

docs/experiment_plan.md

## Ground Rules
1. **Do not use legacy scripts** (`scripts/run_stt.py`, `src/lora/runners/real_small.py`)
   to avoid conflicts with other agents/processes.
2. **Use a Unified Experiment Runner** (`src/lora/runners/experiment.py`) for all hypotheses.
   - Configurable via CLI args (steps, LR, dataset).
   - Prevents code divergence (fixing a bug in H1 fixes it for H3).
3. **Keep evaluation parity**: identical manifest, normalization, and decode
   settings across baseline and tuned runs.

## Hypotheses
1. **Correctness (H1)**: Using `PeftModel` wrappers for both *training* (`model(**batch)`)
   and *evaluation* (`model.generate`) will ensure adapters are actually trained and used.
2. **Volume (H2)**: Increased training volume (≈ 2000 steps on 1000 samples) will produce
   stable WER improvements instead of noisy regressions.
3. **Hyperparameters (H3)**: Learning-rate sweep (1e-4 and 3e-4) will converge faster
   without WER collapse now that normalization is consistent.
4. **Generalization (H4)**: Domain-shift evaluation will show gains where clean
   LibriSpeech is saturated.

## Implementation Plan

### 1. Build Unified Experiment Runner (`src/lora/runners/experiment.py`)
- **Core Logic**:
  - **Training**: Must use `model(**batch)` (wrapped), NOT `base_model(**batch)`.
  - **Eval**: Must use `model.generate(...)` (wrapped) with consistent Moonshine settings.
  - **Normalization**: Must use `normalize_audio_rms(0.075)` in data loading.
- **CLI Arguments**:
  - `--max-steps` (for H2)
  - `--learning-rate` (for H3)
  - `--dataset-path` / `--manifest-path` (for H4)
  - `--eval-interval` (to monitor long runs)

### 2. Execute H1 (Baseline Correctness)
- **Config**: `max_steps=200`, `lr=1e-5` (conservative).
- **Goal**: Confirm `train_loss` drops and `tuned_wer` is at least comparable to baseline
  (no catastrophic failure due to bugs).

### 3. Execute H2 (Long Run)
- **Config**: `max_steps=2000`, `lr=1e-4`, `eval_interval=200`.
- **Validation**: Compute WER on validation subset every 200 steps.
- **Goal**: Clear convergence curve. Stop if WER diverges > 10% from baseline.

### 4. Execute H3 (LR Sweep)
- **Config**: Run 1: `lr=1e-4`, Run 2: `lr=3e-4`.
- **Goal**: Find the highest stable LR.

### 5. Execute H4 (Domain Shift)
- **Config**: Best LR/Steps from above, but evaluated on `heldout_manifest.jsonl`
  (or a new noisy manifest if available).

## Verification Criteria
1. **Code Parity**: Training loop and Eval loop BOTH use `model` (PeftModel), not `base_model`.
2. **Normalization Parity**: RMS normalization applied in training and eval.
3. **Decode Parity**: Moonshine decode settings match across baseline/tuned.
4. **Artifact Integrity**: Runner saves adapter, processor, and metrics.

## Acceptance Criteria
1. Tuned WER < baseline WER on the primary held-out manifest.
2. Improvement reproduces across two runs with fixed seeds.
3. Long runs (H2) show monotonic loss decrease.
4. `docs/experiment_plan.md` is updated with run configs and results.

## Completion Guidance (Do Not Stop Early)
1. **Run, debug, and complete every step end-to-end without stopping.**
2. If a hypothesis fails (e.g. H2 diverges), record the result and move to the next hypothesis/config without reverting to legacy scripts.
3. **Always log exact commands, seeds, and output paths.**
4. Record any fallback device usage or manual adjustments in `notes.md` with the date and reason.
5. **Unblock yourself using best practices** when stuck:
   - Verify manifest parity, normalization parity, and decode parity first.
   - Add targeted logging to isolate where adapter settings diverge.
   - Re-run a minimal, deterministic subset (10–20 samples) to validate the fix.
   - If the runner fails, fix the unified runner (`src/lora/runners/experiment.py`) rather than patching shared scripts.
   - When results are noisy, increase evaluation coverage (more samples) before changing hyperparameters.
