# LoRA Optimization Plan (Updated 2026-02-19)

## Objective
Achieve a stable, reproducible reduction of Domain WER by >= 1.0% using the **Moonshine v2** architecture.

## Phase 1: v2 Integration (Immediate)
- **Model ID Update**: Switch to `UsefulSensors/moonshine-streaming-tiny` to access the Feb 2026 sliding-window architecture.
- **Verification**: Confirm that baseline WER on the expanded 500-sample manifest matches the v2 paper benchmarks (~12.6%).

## Phase 2: The "Bridge-Focus" Strategy
- **Experiment 2.2: Stability Floor**
    - Run 1000 steps at **1e-5 LR** with DoRA to establish a non-regressing baseline for v2.
- **Experiment 2.3: Bridge Fine-Tuning**
    - Target the **Decoder** and **Adapter** stages specifically. 
    - Goal: Use the 24M+ parameters in these stages to map acoustic features to domain vocabulary.

## Phase 3: Validation
- **Checkpointing**: Use `lora_adapter_best` for all final metrics.
- **Normalization**: Maintain strict 0.075 RMS parity.

## Acceptance Criteria
- Domain WER Reduction >= 1.0%
- Heldout WER Regression <= 0.2%
