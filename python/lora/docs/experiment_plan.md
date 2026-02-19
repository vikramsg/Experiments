# Experiment Plan

Single source of truth for active LoRA experiments.

## Active Goals
- **Target Architecture**: **Moonshine v2** (via `UsefulSensors/moonshine-streaming-tiny`).
- **Domain WER**: Reduce absolute WER by >= 1.0% on `data/domain_manifest.jsonl`.
- **Guardrail**: Maintain Heldout WER within 0.2% of baseline.

## Active Protocol
1. **Model ID**: `UsefulSensors/moonshine-streaming-tiny` (Required for v2 architecture).
2. **Training data**: `data/train_manifest_expanded.jsonl` (max-seconds=20).
3. **Evaluation**: Dual-manifest (Domain + Heldout) with Best-WER checkpointing.
4. **Normalization**: Strict RMS 0.075 across all paths.

## Research-Derived Strategy: The "Bridge-Focus" Approach

Analysis of the Moonshine v2 paper and current log failures (Loss dropping but WER flat) indicates that fine-tuning must move beyond the encoder.

### Technical Justification
- **Parameter Density**: The Decoder (22.8M) is 3x larger than the Encoder.
- **The Adapter Stage**: The 1.31M parameter Adapter is the high-leverage target for correcting temporal alignment in domain-shifted audio.
- **Cross-Attention**: Adapting the Decoder's cross-attention is necessary to improve how the model maps acoustic features to domain vocabulary.

### Current Run Matrix

| Run ID | Model | Steps | LR | Target Modules | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| exp_2_3_v2_bridge_focus | streaming-tiny | 1000 | 1e-5 | All Linear | **Running / Flawed.** (Failed to explicitly target Adapter/Decoder due to HF module naming). Baseline: 12.85% (Domain), 4.75% (Heldout). |
| exp_2_4_v2_true_bridge | streaming-tiny | 1000 | 1e-5 | Decoder Attention & MLP | **Planned.** |

### Architectural Correction (2026-02-19)
The Hugging Face implementation of `UsefulSensors/moonshine-streaming-tiny` does not expose an explicit `adapter` module containing linear layers. The position-bridging logic is handled implicitly via `pos_emb` and identity projections. Furthermore, targeting generic linear names (`q_proj`, `fc1`, etc.) targets *both* the Encoder and the Decoder.
To truly execute the "Bridge-Focus" strategy, we must explicitly restrict LoRA targets to the **Decoder's** Cross-Attention (`encoder_attn`) and optionally its self-attention and MLP layers, avoiding the Encoder entirely to force the model to adapt its acoustic-to-vocabulary mapping rather than its fundamental acoustic representations.

## Current Conclusions (v2)
- **Baseline Validated**: Moonshine v2 `streaming-tiny` establishes a stable 12.85% Domain WER floor.
- **Error Profile**: Baseline errors show phonetic confusion on domain terms (e.g., "mester gurr" -> "mr gerk").
- **Stability**: Low LR (1e-5) is currently showing stable loss reduction without immediate WER spikes.

## 2026-02-19 Optimization Plan (Revised v3)

### Objective
Stable reduction of Domain WER (>= 1.0%) with strict Heldout guardrails using v2 architecture.

### Phases
1.  **Phase 1: v2 Infrastructure**: (Complete) Shifted to `streaming-tiny`, filtered out non-Linear targets.
2.  **Phase 2: Bridge-Focus Execution**: (Active) Halt `exp_2_3` and start `exp_2_4_v2_true_bridge`. 1000-step DoRA run at 1e-5 LR explicitly targeting only decoder attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP (`fc1`, `fc2`) layers, omitting the encoder.
3.  **Phase 3: Deep Evaluation**: If Phase 2 succeeds, validate the `lora_adapter_best` and finalize reporting.

### Acceptance Criteria
- Domain WER reduction >= 1.0% (validated on 500-sample stable manifest).
- Heldout WER regression <= 0.2%.
- Zero `Identity()` layer errors.
