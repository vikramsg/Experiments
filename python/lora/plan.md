# LoRA Optimization Plan (Updated 2026-02-19)

## Objective
Achieve a stable, reproducible reduction of Domain WER (`data/domain_manifest.jsonl`) by >= 1.0% absolute, while ensuring Heldout WER (`data/heldout_manifest.jsonl`) remains within a 0.2% guardrail.

## Phase 1: Infrastructure & Data Stability (Immediate)
- **Task 1.1: Runner Enhancements**
    - Implement **Best-Checkpoint Saving**: Track WER at every interval and save the best performing adapter to `lora_adapter_best`.
    - Implement **LR Scheduler**: Add a Linear Warmup (10% of steps) and Linear Decay scheduler to prevent early divergence and facilitate convergence.
- **Task 1.2: Manifest Expansion**
    - Generate an expanded domain evaluation manifest (500+ samples) to mitigate the "Lucky Average" bias where the final 50 samples of the current 200-sample set are disproportionately easy.
- **Task 1.3: Normalization Parity**
    - Ensure all inference, training, and evaluation paths strictly apply RMS normalization (target 0.075).

## Phase 2: The "Long-and-Slow" DoRA Strategy
- **Experiment 2.1: DoRA Stability Run**
    - Config: 1000 steps, LR 1e-4, use DoRA, Linear Decay.
    - Goal: Complete ~3 full epochs over the expanded training data to allow the adapter to actually learn the domain distribution.
- **Experiment 2.2: Architecture Refinement**
    - If 2.1 is stable but slow, evaluate limiting `target_modules` to `["q_proj", "v_proj"]` vs. full attention/MLP to reduce parameter noise.

## Phase 3: Hyperparameter Search
- **Experiment 3.1: Learning Rate Sweep (Refined)**
    - Test LR [5e-5, 1e-4, 1.5e-4] with the new scheduler.
    - **Note**: Avoid LRs >= 2e-4 as they have demonstrated catastrophic forgetting in previous runs.

## Verification Criteria
1. **Convergence**: Evaluation loss must show a downward trend before WER is considered valid.
2. **Stability**: The WER across the last 3 intervals of a 1000-step run should not fluctuate by more than 0.5% absolute.
3. **Artifact Integrity**: `lora_adapter_best` must be used for all final "Tuned WER" reporting.

## Acceptance Criteria
- **Domain WER**: Absolute reduction >= 1.0%.
- **Heldout WER**: Absolute regression <= 0.2%.
- **Parity**: The same performance must be reproducible in the `lora-cli` using identical normalization.

## Execution Guidance
**Do not stop until the acceptance criteria are met.**
Previous failures (exp_1.1, exp_2.1b) have proven that short runs and high learning rates are unstable. Persistence through longer, scheduled runs is the only path to the target. Every result must be documented in the Run Matrix.
