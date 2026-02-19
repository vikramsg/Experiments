# LoRA Optimization Plan (Updated 2026-02-19 - v2)

## Objective
Achieve a stable, reproducible reduction of Domain WER (`data/domain_manifest.jsonl`) by >= 1.0% absolute, while ensuring Heldout WER (`data/heldout_manifest.jsonl`) remains within a 0.2% guardrail.

## Phase 1: Infrastructure & Data Stability (COMPLETE)
- [x] **Runner Enhancements**: Implemented LR scheduler, best-checkpoint saving, and dual-manifest evaluation.
- [x] **Manifest Expansion**: Expanded domain manifest to 500 samples for statistical stability.
- [x] **Normalization Parity**: RMS 0.075 normalization confirmed across paths.

## Phase 2: The "Conservative" DoRA Strategy (Active)
- **Experiment 2.2: Low-and-Slow DoRA**
    - **Diagnosis**: Previous runs at 1e-4 LR showed catastrophic forgetting (rising WER despite falling loss).
    - **Config**: 1000 steps, **LR 1e-5**, 200-step warmup (20%), use DoRA.
    - **Goal**: Gradual adaptation that preserves pre-trained linguistic knowledge while aligning with domain acoustics.
- **Experiment 2.3: Target Module Narrowing (Contingency)**
    - If 2.2 is stable but lacks "bite," limit LoRA to `["q_proj", "v_proj"]` to reduce the degrees of freedom for forgetting.

## Phase 3: Final Optimization & Validation
- **Experiment 3.1: Micro-LR Sweep**
    - Test LR [8e-6, 1e-5, 3e-5] around the new stable floor.
- **Verification**: Final validation of `lora_adapter_best` on both manifests.

## Acceptance Criteria
- **Domain WER**: Absolute reduction >= 1.0%.
- **Heldout WER**: Absolute regression <= 0.2%.
- **Stability**: No upward "diving" or "spiking" in WER during the final 400 steps.

## Execution Guidance
**I will operate with full autonomy and will not stop until these criteria are met.** Any further regressions will be met with immediate strategy adjustments (lower LR, higher weight decay, or architectural narrowing) until the target is hit.
