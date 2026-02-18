# Plan: LoRA Tuning Experiments (Hypothesis-Driven)

## Goal
Establish a reproducible tuning workflow that improves WER on target-domain data
while keeping baseline performance stable, using hypothesis-specific runners
instead of shared scripts.

## Reference

docs/experiment_plan.md

## Ground Rules
1. **Do not use existing scripts** (`scripts/run_stt.py`, `src/lora/runners/real_small.py`, etc.)
   because they are being used by another agent.
2. **Create hypothesis-specific runners** for each experiment (one runner per
   hypothesis), so runs are isolated and easy to reproduce.
3. **Keep evaluation parity**: identical manifest, normalization, and decode
   settings across baseline and tuned runs.

## Hypotheses
1. Adapter-aware evaluation (use `PeftModel.generate`) will align in-training
   WER with final STT WER.
2. Increased training volume (≈ 2000 steps on 1000 samples) will produce
   stable WER improvements instead of noisy regressions.
3. Learning-rate sweep (1e-4 and 3e-4) will converge faster without WER collapse
   now that normalization is consistent.
4. Domain-shift evaluation will show gains where clean LibriSpeech is saturated.

## Implementation Plan (Hypothesis Runners)

### 1. Build Dedicated Runner Templates
- Create a minimal runner skeleton per hypothesis (e.g.,
  `src/lora/runners/h1_eval_adapter.py`, `h2_long_run.py`, etc.).
- Each runner must:
  - define its own CLI args
  - log a full config summary
  - save artifacts to a hypothesis-specific output directory

### 2. Adapter-Aware Evaluation Runner (H1)
- Ensure evaluation uses `model.generate(...)` when adapters are active.
- Compare in-training WER against a held-out manifest using the same settings.

### 3. Long-Run Convergence Runner (H2)
- Run ~2000 steps on 1000 samples, batch size 1, grad_accum 4.
- Capture WER every N updates to detect convergence/overfitting.

### 4. Learning Rate Sweep Runner (H3)
- Two runs: 1e-4 and 3e-4, same steps and data.
- Compare tuned WER vs baseline on the same manifest.

### 5. Domain-Shift Evaluation Runner (H4)
- Evaluate tuned adapters on a harder/noisy manifest.
- Report WER delta vs baseline using the same decode settings.

## Verification Criteria
1. **Adapter parity**: evaluation calls use adapter-aware `generate`.
2. **Normalization parity**: RMS normalization applied in training and eval.
3. **Decode parity**: Moonshine decode settings match across baseline/tuned.
4. **Manifest parity**: baseline/tuned metrics computed on identical manifests.
5. **Artifact integrity**: each runner saves adapter, processor, and metrics.

## Acceptance Criteria
1. Tuned WER < baseline WER on the primary held-out manifest.
2. Tuned WER is non-worse on at least one additional manifest.
3. Improvement reproduces across two runs with fixed seeds.
4. All hypothesis-specific runners complete without runtime failures.
5. `docs/experiment_plan.md` and `docs/training_report.md` are updated with
   run configs, metrics, and artifact paths.

## Completion Guidance (Do Not Stop Early)
1. Run, debug, and complete every step end-to-end without stopping. Unblock yourself using best practices 
2. If a hypothesis fails, record the result and move to the next hypothesis
   runner without reusing the shared scripts.
3. Always log exact commands, seeds, and output paths.
4. Record any fallback device usage or manual adjustments in `notes.md` with
   the date and reason.
5. **Unblock yourself using best practices** when stuck:
   - Verify manifest parity, normalization parity, and decode parity first.
   - Add targeted logging to isolate where adapter settings diverge.
   - Re-run a minimal, deterministic subset (10–20 samples) to validate the fix.
   - If a runner fails, create a new hypothesis runner that isolates the failure
     rather than patching shared scripts.
   - When results are noisy, increase evaluation coverage before changing
     hyperparameters.
