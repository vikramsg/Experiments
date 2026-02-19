# LoRA Optimization Plan

## Objective
Significant reduction of Domain WER (`data/domain_manifest.jsonl`) while maintaining Heldout WER (`data/heldout_manifest.jsonl`) within a tight guardrail.

## Phase 1: Preprocessing & Baseline Alignment
- **Task**: Align audio preprocessing between training and inference.
    - Audit `src/lora/data_loader.py` and `packages/lora-cli/src/lora_cli/audio.py`.
    - Ensure RMS normalization is identical in both paths.
- **Experiment 1.1**: Re-run Domain Baseline (Standard LoRA, LR 1e-4, 200 steps) with aligned preprocessing.
- **Verification**: Confirm that the baseline WER matches expectations and the "tuned" improvement is not masked by preprocessing drift.

## Phase 2: Architectural Advancement
- **Experiment 2.1: DoRA (Weight-Decomposed LoRA)**
    - Enable `use_dora=True` in `LoraConfig`.
    - Goal: Test if decoupling magnitude and direction improves convergence on domain data.
- **Experiment 2.2: PiSSA Initialization**
    - Enable `init_lora_weights="pissa"`.
    - Goal: Test if principal singular value initialization provides a better starting point for the adapter.

## Phase 3: Hyperparameter Optimization (The "Sweep")
- **Experiment 3.1: Learning Rate Sweep**
    - Test LR values: [1e-5, 5e-5, 1e-4, 3e-4] using the best architecture from Phase 2.
- **Experiment 3.2: Step/Duration Sweep**
    - Increase max steps to [500, 1000] with a linear decay scheduler.
    - Goal: Utilize the full expanded dataset (`data/train_manifest_expanded.jsonl`) without hitting the regressions seen in earlier long runs.

## Verification Criteria
1. **Parity Check**: Baseline WER on `domain_manifest` must be reproducible within +/- 0.1% before proceeding to tuning.
2. **Reproducibility**: Every run must have a corresponding entry in `docs/experiment_plan.md` with the exact `uv run` command used.

## Acceptance Criteria
- **Primary**: Absolute reduction in Domain WER of at least **1.0%** (e.g., 12.5% -> 11.5%).
- **Guardrail**: Absolute regression in Heldout WER must be less than **0.2%** (e.g., 4.49% -> 4.69% max).
- **Consistency**: The model must not show signs of "catastrophic forgetting" on general audio.

## Execution Guidance
**Do not stop until all acceptance criteria are met.**
The task is not complete upon running the scripts; it is complete when a hyperparameter set is found that breaks the current performance plateau. Every failure is a data point—document why a specific LR or architecture regressed before moving to the next.
