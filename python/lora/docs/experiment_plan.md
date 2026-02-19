# Experiment Plan

Single source of truth for active LoRA experiments.

Legacy/incorrect workflow results were archived to:
- `docs/archive_legacy_results.md`

## Active Goals
- Improve WER on domain-shifted audio (`data/domain_manifest.jsonl`).
- Keep heldout performance non-regressing (`data/heldout_manifest.jsonl`).
- Use reproducible settings that avoid data starvation.

## Active Protocol (Current)
1. **Training data**: `data/train_manifest_expanded.jsonl`
2. **Duration filter**: `--max-seconds=20`
3. **Evaluation manifests**:
   - primary: `data/domain_manifest.jsonl` (Targeting 500+ samples for stability)
   - safety: `data/heldout_manifest.jsonl`
4. **Normalization/decode parity**: Strict RMS normalization (0.075) across all paths.
5. **Adapter parity**: Evaluate with active adapters (**No unwrap_peft**).
6. **Checkpointing**: Always use the **Best-WER checkpoint** for final reporting.

## Current Run Matrix (Retained)

| Run ID | Date | Command | Dataset | Steps | LR | Manifest | Baseline WER | Tuned WER | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h1-expanded-20260218 | 2026-02-18 | `uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_h1_expanded_20260218 --max-steps 200 --learning-rate 1e-5 --eval-interval 50 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/heldout_manifest.jsonl --max-seconds 20 --seed 42` | `data/train_manifest_expanded.jsonl` | 200 | 1e-5 | `data/heldout_manifest.jsonl` | 0.0449 | 0.0449 | Flat heldout WER. |
| h2-expanded-20260218 | 2026-02-18 | `uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_h2_expanded_20260218 --max-steps 2000 --learning-rate 1e-4 --eval-interval 200 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/heldout_manifest.jsonl --max-seconds 20 --seed 42 --wer-stop-threshold 1.1` | `data/train_manifest_expanded.jsonl` | 2000 | 1e-4 | `data/heldout_manifest.jsonl` | 0.0449 | 0.0552 | Regressed; threshold stop hit. |
| h3-lr1e4-expanded-20260218 | 2026-02-18 | `uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_h3_lr1e4_expanded_20260218 --max-steps 200 --learning-rate 1e-4 --eval-interval 50 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/heldout_manifest.jsonl --max-seconds 20 --seed 42` | `data/train_manifest_expanded.jsonl` | 200 | 1e-4 | `data/heldout_manifest.jsonl` | 0.0449 | 0.0475 | Slight regression. |
| h3-lr3e4-expanded-20260218 | 2026-02-18 | `uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_h3_lr3e4_expanded_20260218 --max-steps 200 --learning-rate 3e-4 --eval-interval 50 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/heldout_manifest.jsonl --max-seconds 20 --seed 42` | `data/train_manifest_expanded.jsonl` | 200 | 3e-4 | `data/heldout_manifest.jsonl` | 0.0449 | 0.0723 | Regressed at higher LR. |
| h4-domain-expanded-20260218 | 2026-02-18 | `uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_h4_domain_expanded_20260218 --max-steps 200 --learning-rate 1e-4 --eval-interval 50 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/domain_manifest.jsonl --max-seconds 20 --seed 42` | `data/train_manifest_expanded.jsonl` | 200 | 1e-4 | `data/domain_manifest.jsonl` | 0.1168 | 0.1150 | Small domain improvement. |
| headroom-heldout-20260218 | 2026-02-18 | `uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_headroom_heldout_20260218 --max-steps 100 --learning-rate 1e-4 --eval-interval 50 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/heldout_manifest.jsonl --max-seconds 20 --seed 42` | `data/train_manifest_expanded.jsonl` | 100 | 1e-4 | `data/heldout_manifest.jsonl` | 0.0449 | 0.0458 | Heldout remains headroom-limited. |
| headroom-domain-20260218 | 2026-02-18 | `uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_headroom_domain_20260218 --max-steps 100 --learning-rate 1e-4 --eval-interval 50 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/domain_manifest.jsonl --max-seconds 20 --seed 42` | `data/train_manifest_expanded.jsonl` | 100 | 1e-4 | `data/domain_manifest.jsonl` | 0.1261 | 0.1253 | Domain headroom confirmed. |
| exp_1_1_baseline_fixed | 2026-02-19 | `uv run python src/lora/runners/experiment.py --output-dir outputs/exp_1_1_baseline_fixed --max-steps 200 --learning-rate 1e-4 --eval-interval 50 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/domain_manifest.jsonl --max-seconds 20 --seed 42` | `data/train_manifest_expanded.jsonl` | 200 | 1e-4 | `data/domain_manifest.jsonl` | 0.1168 | 0.1349 | **Regression.** Fixed eval bug reveals LoRA instability at 1e-4. |
| exp_2_1_dora_base | 2026-02-19 | `uv run python src/lora/runners/experiment.py --output-dir outputs/exp_2_1_dora_base --max-steps 200 --learning-rate 5e-5 --eval-interval 50 --use-dora --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/domain_manifest.jsonl --max-seconds 20 --seed 42` | `data/train_manifest_expanded.jsonl` | 200 | 5e-5 | `data/domain_manifest.jsonl` | 0.1174 | 0.1184 | **Stable.** DoRA prevents regression at 5e-5. |
| exp_2_1b_dora_high_lr | 2026-02-19 | `uv run python src/lora/runners/experiment.py --output-dir outputs/exp_2_1b_dora_high_lr --max-steps 200 --learning-rate 2e-4 --eval-interval 50 --use-dora --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/domain_manifest.jsonl --max-seconds 20 --seed 42` | `data/train_manifest_expanded.jsonl` | 200 | 2e-4 | `data/domain_manifest.jsonl` | 0.1174 | 0.2241 | **Severe Regression.** 2e-4 is too high even for DoRA. |

## Current Conclusions
- Expanded manifest + 20s filtering removed the earlier data starvation issue.
- Heldout remains low-headroom and should be treated as a guardrail metric.
- Domain manifest has sufficient headroom and is the right optimization target.

## Guardrail Note (Incorrect Setup)
- The old `--max-seconds=8` workflow collapsed training data to near-empty in this repo and produced unstable/noisy WER behavior.
- Full evidence and old run logs are documented in `docs/archive_legacy_results.md`.

## 2026-02-19 Optimization Plan (Revised)

### Objective
Stable reduction of Domain WER (>= 1.0%) with strict Heldout guardrails.

### Phases
1.  **Phase 1: Infrastructure**: Add LR schedulers and Best-Checkpoint saving to the runner.
2.  **Phase 2: Data Stability**: Expand domain manifest to 500+ samples to remove "Lucky Average" bias.
3.  **Phase 3: DoRA Deep-Dive**: Execute "Long-and-Slow" runs (1000 steps) with scheduled LR.

### Acceptance Criteria
- Domain WER reduction >= 1.0% (validated on stable manifest).
- Heldout WER regression <= 0.2%.
- Full documentation of all intermediate failures and sweeps.

### Guidance
Persistence is mandatory. Continue iterating on hyperparameters and architectures until the acceptance criteria are met. Document all intermediate results in the Run Matrix.
