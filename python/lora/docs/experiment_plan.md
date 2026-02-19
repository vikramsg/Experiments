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

## Current Run Matrix

| Run ID | Model | Steps | LR | Target Modules | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| exp_2_3_v2_bridge_focus | streaming-tiny | 1000 | 1e-5 | All Linear | **Flawed.** (Failed to explicitly target Adapter/Decoder due to HF module naming). |
| exp_2_4_v2_true_bridge | streaming-tiny | 1000 | 1e-5 | Decoder Attention & MLP (DoRA) | **Failed.** (Regressed on Domain and Heldout). |
| exp_2_5_v2_universal_lora | streaming-tiny | 1000 | 1e-5 | Global `q,k,v,o_proj` (Standard LoRA) | **Planned.** |

## Research & Optimization History

### The "Bridge-Focus" Failure (2026-02-19)
Analysis of the Moonshine v2 paper originally suggested targeting the Decoder (22.8M) and Adapter (1.31M) while freezing the streaming position-free Encoder. `exp_2_4` attempted to execute this by applying DoRA with `LORA_MODULE_FILTER="decoder"`.
**Result:** The model steadily regressed. Domain WER increased from a 12.40% baseline to 12.64% by step 600, and Heldout regressed from 4.79% to 4.92%.
**Conclusion:** Restricting LoRA explicitly to the `decoder` module in the Hugging Face implementation likely misses critical implicit adapter projection layers that bridge the positional gap. Alternatively, DoRA (Weight-Decomposed LoRA) is proving too unstable at `1e-5` LR on this specific 33M parameter architecture.

## 2026-02-19 Universal Baseline Correction (Revised v4)

### Hypothesis
To definitively prove that the v2 architecture can be fine-tuned without degrading, we must abandon the strict architectural filtering that induced regressions in `exp_2_4`. Applying standard universal LoRA (targeting `q_proj`, `k_proj`, `v_proj`, `o_proj` across the *entire* model, both encoder and decoder) and disabling DoRA decomposition will successfully adapt the v2 model to the domain. This approach relies on the proven, generic LoRA setup that previously yielded successful adaptations on the v1 model.

### Phases
1.  **Phase 1: v2 Infrastructure**: (Complete) Shifted to `streaming-tiny`.
2.  **Phase 2: Universal Baseline Execution**: (Active) Halt `exp_2_4`. Start `exp_2_5_v2_universal_lora`. Run a 1000-step standard LoRA training at `1e-5` LR. Remove the `LORA_MODULE_FILTER` environment variable and the `--use-dora` flag, targeting global attention projections (`q_proj, k_proj, v_proj, o_proj`).
3.  **Phase 3: Deep Evaluation**: If Phase 2 succeeds in preventing regression, validate the `lora_adapter_best` and establish it as the true v2 fine-tuning baseline.

### Acceptance Criteria
- Domain WER reduction >= 1.0% (validated on 500-sample stable manifest).
- Heldout WER regression <= 0.2%.
- Zero `Identity()` layer errors.