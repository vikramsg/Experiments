# LoRA Optimization Plan (Updated 2026-02-19)

## Objective
Achieve a stable, reproducible reduction of Domain WER by >= 1.0% using the **Moonshine v2** architecture.

## Phase 1: v2 Integration (Complete)
- **Model ID Update**: Switch to `UsefulSensors/moonshine-streaming-tiny` to access the Feb 2026 sliding-window architecture.
- **Verification**: Confirm that baseline WER on the expanded 500-sample manifest matches the v2 paper benchmarks (~12.6%).

## Phase 2: Universal LoRA Baseline (Active)
- **Hypothesis**: The previous "Bridge-Focus" strategy (restricting DoRA strictly to the decoder) destabilizes the v2 model, leading to clear regressions. This is likely because the HF implementation's implicit adapter/projection layers are completely missed by strict `LORA_MODULE_FILTER="decoder"` filtering, or DoRA decomposition is too unstable at 1e-5 LR for this specific 33M parameter model. A universal LoRA application (targeting `q_proj`, `k_proj`, `v_proj`, `o_proj` across the *entire* model, both encoder and decoder) using standard LoRA will successfully adapt the v2 model to the domain without regressing, mirroring the historical baseline success seen on the v1 architecture.
- **Experiment 2.5: Universal Baseline Setup**
    - Halt `exp_2_4_v2_true_bridge` (actively regressing on Domain and Heldout).
    - Start `exp_2_5_v2_universal_lora`.
    - Run 1000 steps at **1e-5 LR** with standard LoRA (remove `--use-dora`).
    - Remove `LORA_MODULE_FILTER`. Target standard attention projections globally (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
    - Goal: Prove that the v2 architecture *can* be fine-tuned without regressing on heldout data, establishing a valid baseline for future targeted optimizations.

## Phase 3: Validation
- **Checkpointing**: Use `lora_adapter_best` for all final metrics.
- **Normalization**: Maintain strict 0.075 RMS parity.

## Acceptance Criteria
- Domain WER Reduction >= 1.0%
- Heldout WER Regression <= 0.2%