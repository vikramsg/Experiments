# Experiment Plan

Single source of truth for what we want to test, why, and how we decide
whether it worked. Keep this aligned with `docs/training.md` guidance and
reference concrete run outputs in `docs/training_report.md`.

## Goals
- Improve WER on target-domain audio while keeping baseline performance stable.
- Avoid train/inference mismatch (normalization + decode settings).
- Ensure evaluation calls respect active LoRA adapters.
- Produce reproducible improvements across at least two runs.

## Evaluation Protocol (Do Not Deviate)
1. **Manifest parity**: baseline and tuned WER must use the same manifest.
2. **Normalization parity**: RMS normalization must match inference behavior.
3. **Decode parity**: use the same Moonshine decode settings everywhere.
4. **Adapter parity**: evaluation must call adapter-aware `generate` when LoRA is active.
5. **Coverage**: use full-manifest WER for primary decisions; tiny subsets are POC only.

## Known Risks / Failure Modes
- **PEFT bypass in eval**: `unwrap_peft(model)` uses `base_model.generate`, which can
  skip adapter-aware generation when adapters are active.
- **Clean dataset saturation**: `librispeech_asr` clean manifests have baseline WER ≈ 4–5%,
  leaving little headroom for improvement.
- **Loss ≠ WER**: improving loss can still worsen WER if decoding is misaligned.
- **Small-run noise**: short runs can regress WER even with stable loss curves.

## Decision Criteria
- **Primary**: tuned WER < baseline WER on the main heldout manifest.
- **Secondary**: tuned WER is non-worse on at least one additional manifest.
- **Reproducibility**: same improvement holds for two runs with fixed seeds.
- **Sanity**: training loss decreases without catastrophic WER regression (>2×).

## Hypotheses and Tests

### H1 — Adapter-aware Generation in Evaluation
- **Problem**: `src/lora/evaluation.py` unwraps PEFT and calls `base_model.generate`.
- **Risk**: LoRA adapter configuration may be ignored during evaluation.
- **Change**: call `model.generate` (PeftModel) when adapters are active.
- **Expected**: in-training WER aligns with `scripts/run_stt.py` WER direction.
- **Success**: tuned WER from training eval matches the trend from `run_stt.py`.

### H2 — Training Volume for 1k Samples
- **Problem**: 50–200 updates are unlikely to converge; WER is noisy.
- **Change**: use ~2000 steps (batch 1, grad_accum 4, 1000 samples ≈ 8 epochs).
- **Expected**: clearer convergence/overfitting signal; WER trend stabilizes.
- **Success**: tuned WER improves or overfits in a predictable monotonic trend.

### H3 — Learning Rate Revisit Post-Normalization
- **Problem**: prior regression may have been normalization mismatch, not LR.
- **Change**: test 1e-4 and 3e-4 with normalization + decode fixed.
- **Expected**: faster loss convergence without WER collapse.
- **Success**: tuned WER improves or stays within ±5% of baseline.

### H4 — Harder / Domain-Shifted Evaluation
- **Problem**: clean LibriSpeech is near model capacity; limited headroom.
- **Change**: add a noisy/accents/domain manifest.
- **Expected**: tuned WER improves on domain shift even if clean stays flat.
- **Success**: tuned WER improves on domain manifest while clean stays non-worse.

## Test Matrix (Fill Per Run)
| Run ID | Date | Command | Dataset | Steps | LR | LoRA Targets | Manifest | Baseline WER | Tuned WER | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h1-20260218 | 2026-02-18 | uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_h1_20260218 --max-steps 200 --learning-rate 1e-5 --eval-interval 50 --dataset-path data/train_manifest.jsonl --manifest-path data/heldout_manifest.jsonl --seed 42 | data/train_manifest.jsonl | 200 | 1e-5 | q_proj,k_proj,v_proj,fc1,fc2 | data/heldout_manifest.jsonl | 0.0462 | 0.0462 | Interval WER stable at 0.0462; loss down to 9.45. |
| h2-20260218 | 2026-02-18 | uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_h2_20260218 --max-steps 2000 --learning-rate 1e-4 --eval-interval 200 --dataset-path data/train_manifest.jsonl --manifest-path data/heldout_manifest.jsonl --seed 42 --wer-stop-threshold 1.1 | data/train_manifest.jsonl | 2000 | 1e-4 | q_proj,k_proj,v_proj,fc1,fc2 | data/heldout_manifest.jsonl | 0.0462 | 0.0582 | WER regressed >10% at step 400; stopped after threshold; loss improved. |
| h3-lr1e4-20260218 | 2026-02-18 | uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_h3_lr1e4_20260218 --max-steps 200 --learning-rate 1e-4 --eval-interval 50 --dataset-path data/train_manifest.jsonl --manifest-path data/heldout_manifest.jsonl --seed 42 | data/train_manifest.jsonl | 200 | 1e-4 | q_proj,k_proj,v_proj,fc1,fc2 | data/heldout_manifest.jsonl | 0.0462 | 0.0462 | Interval WER briefly 0.0479 then recovered; tuned WER unchanged. |
| h3-lr3e4-20260218 | 2026-02-18 | uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_h3_lr3e4_20260218 --max-steps 200 --learning-rate 3e-4 --eval-interval 50 --dataset-path data/train_manifest.jsonl --manifest-path data/heldout_manifest.jsonl --seed 42 | data/train_manifest.jsonl | 200 | 3e-4 | q_proj,k_proj,v_proj,fc1,fc2 | data/heldout_manifest.jsonl | 0.0462 | 0.0701 | WER regressed; loss dropped to 4.37. |
| h4-20260218 | 2026-02-18 | uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_h4_20260218 --max-steps 200 --learning-rate 1e-4 --eval-interval 50 --dataset-path data/train_manifest.jsonl --manifest-path data/test_manifest.jsonl --seed 42 | data/train_manifest.jsonl | 200 | 1e-4 | q_proj,k_proj,v_proj,fc1,fc2 | data/test_manifest.jsonl | 0.0000 | 0.0000 | Domain manifest had 2 samples; WER stayed 0.0. |
| h1-expanded-20260218 | 2026-02-18 | uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_h1_expanded_20260218 --max-steps 200 --learning-rate 1e-5 --eval-interval 50 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/heldout_manifest.jsonl --max-seconds 20 --seed 42 | data/train_manifest_expanded.jsonl | 200 | 1e-5 | q_proj,k_proj,v_proj,o_proj,fc1,fc2 | data/heldout_manifest.jsonl | 0.0449 | 0.0449 | Expanded manifest; WER flat; loss 9.63. |
| h2-expanded-20260218 | 2026-02-18 | uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_h2_expanded_20260218 --max-steps 2000 --learning-rate 1e-4 --eval-interval 200 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/heldout_manifest.jsonl --max-seconds 20 --seed 42 --wer-stop-threshold 1.1 | data/train_manifest_expanded.jsonl | 2000 | 1e-4 | q_proj,k_proj,v_proj,o_proj,fc1,fc2 | data/heldout_manifest.jsonl | 0.0449 | 0.0552 | WER regressed >10% at step 400; stopped after threshold. |
| h3-lr1e4-expanded-20260218 | 2026-02-18 | uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_h3_lr1e4_expanded_20260218 --max-steps 200 --learning-rate 1e-4 --eval-interval 50 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/heldout_manifest.jsonl --max-seconds 20 --seed 42 | data/train_manifest_expanded.jsonl | 200 | 1e-4 | q_proj,k_proj,v_proj,o_proj,fc1,fc2 | data/heldout_manifest.jsonl | 0.0449 | 0.0475 | WER regressed slightly; LoRA delta 1281.82. |
| h3-lr3e4-expanded-20260218 | 2026-02-18 | uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_h3_lr3e4_expanded_20260218 --max-steps 200 --learning-rate 3e-4 --eval-interval 50 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/heldout_manifest.jsonl --max-seconds 20 --seed 42 | data/train_manifest_expanded.jsonl | 200 | 3e-4 | q_proj,k_proj,v_proj,o_proj,fc1,fc2 | data/heldout_manifest.jsonl | 0.0449 | 0.0723 | WER regressed; long WER eval stalls on MPS. |
| h4-domain-expanded-20260218 | 2026-02-18 | uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_h4_domain_expanded_20260218 --max-steps 200 --learning-rate 1e-4 --eval-interval 50 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/domain_manifest.jsonl --max-seconds 20 --seed 42 | data/train_manifest_expanded.jsonl | 200 | 1e-4 | q_proj,k_proj,v_proj,o_proj,fc1,fc2 | data/domain_manifest.jsonl | 0.1168 | 0.1150 | Domain WER improved slightly; long WER eval stalls on MPS. |
| headroom-heldout-20260218 | 2026-02-18 | uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_headroom_heldout_20260218 --max-steps 100 --learning-rate 1e-4 --eval-interval 50 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/heldout_manifest.jsonl --max-seconds 20 --seed 42 | data/train_manifest_expanded.jsonl | 100 | 1e-4 | q_proj,k_proj,v_proj,o_proj,fc1,fc2 | data/heldout_manifest.jsonl | 0.0449 | 0.0458 | Expanded train manifest; LoRA delta 592.80; WER flat. |
| headroom-domain-20260218 | 2026-02-18 | uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_headroom_domain_20260218 --max-steps 100 --learning-rate 1e-4 --eval-interval 50 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/domain_manifest.jsonl --max-seconds 20 --seed 42 | data/train_manifest_expanded.jsonl | 100 | 1e-4 | q_proj,k_proj,v_proj,o_proj,fc1,fc2 | data/domain_manifest.jsonl | 0.1261 | 0.1253 | Domain headroom confirmed; LoRA delta 592.80; WER slight improvement. |

## Run Notes
- Record any non-standard action, fallback, or manual intervention with a date.

## Follow-up Diagnostics (Why These Experiments Are Required)

The latest H1–H4 runs show loss improving while WER stays flat or regresses. Before
changing hyperparameters again, we need to confirm that the data pipeline is
providing enough signal, that evaluation headroom exists, and that adapters are
actually affecting inference. These diagnostics are required to avoid tuning
against noise or a saturated benchmark.

### Checklist
- [x] **Manifest duration profile**: Count how many samples survive each `max_seconds` threshold (5/8/10/15/20s).
- [x] **Training split sanity**: Confirm train/val sizes and speaker split logic after filtering.
- [x] **Baseline headroom**: Report baseline WER on heldout + domain manifests; flag if < 5%.
- [x] **Adapter efficacy**: Verify LoRA weight deltas and confirm adapter-aware `generate` is used.
- [x] **Data sufficiency target**: Set minimum viable sample count (e.g., 200–500) before next LR sweep.

### Diagnostics tooling
- Use `uv run python scripts/manifest_diagnostics.py --manifests data/train_manifest.jsonl data/heldout_manifest.jsonl data/test_manifest.jsonl --max-seconds 8` to log duration counts and split sizes.
- Use `uv run python scripts/manifest_diagnostics.py --manifests data/train_manifest_expanded.jsonl data/heldout_manifest.jsonl data/domain_manifest.jsonl --max-seconds 20` for expanded training/domain manifests.
- Adjust `--max-seconds` and `--thresholds` when evaluating new manifests.

## Diagnostics Results (2026-02-18)

### Original manifest duration profile (max-seconds=8)

| Manifest | Samples | Mean (s) | <=5s | <=8s | <=10s | <=15s | <=20s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `data/train_manifest.jsonl` | 120 | 13.33 | 6 (5.0%) | 8 (6.7%) | 17 (14.2%) | 85 (70.8%) | 120 (100.0%) |
| `data/heldout_manifest.jsonl` | 120 | 6.94 | 50 (41.7%) | 85 (70.8%) | 96 (80.0%) | 114 (95.0%) | 119 (99.2%) |
| `data/test_manifest.jsonl` | 2 | 6.87 | 0 (0.0%) | 2 (100.0%) | 2 (100.0%) | 2 (100.0%) | 2 (100.0%) |

### Original split sizes (max-seconds=8, seed=42)

| Manifest | Filtered | Train | Val | Split | Source |
| --- | --- | --- | --- | --- | --- |
| `data/train_manifest.jsonl` | 8 | 7 | 1 | random | `uv run python scripts/manifest_diagnostics.py --manifests data/train_manifest.jsonl data/heldout_manifest.jsonl data/test_manifest.jsonl --max-seconds 8` |
| `data/heldout_manifest.jsonl` | 85 | 68 | 17 | random | `uv run python scripts/manifest_diagnostics.py --manifests data/train_manifest.jsonl data/heldout_manifest.jsonl data/test_manifest.jsonl --max-seconds 8` |
| `data/test_manifest.jsonl` | 2 | 1 | 1 | random | `uv run python scripts/manifest_diagnostics.py --manifests data/train_manifest.jsonl data/heldout_manifest.jsonl data/test_manifest.jsonl --max-seconds 8` |

### Expanded manifest duration profile (max-seconds=20)

| Manifest | Samples | Mean (s) | <=5s | <=8s | <=10s | <=15s | <=20s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `data/train_manifest_expanded.jsonl` | 360 | 11.20 | 62 (17.2%) | 101 (28.1%) | 130 (36.1%) | 284 (78.9%) | 359 (99.7%) |
| `data/heldout_manifest.jsonl` | 120 | 6.94 | 50 (41.7%) | 85 (70.8%) | 96 (80.0%) | 114 (95.0%) | 119 (99.2%) |
| `data/domain_manifest.jsonl` | 200 | 9.58 | 48 (24.0%) | 80 (40.0%) | 100 (50.0%) | 171 (85.5%) | 200 (100.0%) |

### Expanded split sizes (max-seconds=20, seed=42)

| Manifest | Filtered | Train | Val | Split | Source |
| --- | --- | --- | --- | --- | --- |
| `data/train_manifest_expanded.jsonl` | 359 | 268 | 77 | speaker | `uv run python scripts/manifest_diagnostics.py --manifests data/train_manifest_expanded.jsonl data/heldout_manifest.jsonl data/domain_manifest.jsonl --max-seconds 20` |
| `data/heldout_manifest.jsonl` | 119 | 96 | 23 | random | `uv run python scripts/manifest_diagnostics.py --manifests data/train_manifest_expanded.jsonl data/heldout_manifest.jsonl data/domain_manifest.jsonl --max-seconds 20` |
| `data/domain_manifest.jsonl` | 200 | 160 | 40 | random | `uv run python scripts/manifest_diagnostics.py --manifests data/train_manifest_expanded.jsonl data/heldout_manifest.jsonl data/domain_manifest.jsonl --max-seconds 20` |

### Headroom runs (max-seconds=20)

| Run | Command | Baseline WER | Tuned WER | Output |
| --- | --- | --- | --- | --- |
| Heldout | `uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_headroom_heldout_20260218 --max-steps 100 --learning-rate 1e-4 --eval-interval 50 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/heldout_manifest.jsonl --max-seconds 20 --seed 42` | 0.0449 | 0.0458 | `outputs/experiment_headroom_heldout_20260218/experiment_metrics.json` |
| Domain | `uv run python src/lora/runners/experiment.py --output-dir outputs/experiment_headroom_domain_20260218 --max-steps 100 --learning-rate 1e-4 --eval-interval 50 --dataset-path data/train_manifest_expanded.jsonl --manifest-path data/domain_manifest.jsonl --max-seconds 20 --seed 42` | 0.1261 | 0.1253 | `outputs/experiment_headroom_domain_20260218/experiment_metrics.json` |

### Key findings
- **Data sufficiency achieved**: `--max-seconds=20` yields 359 filtered samples and a 268/77 train/val split.
- **Domain headroom confirmed**: baseline WER is 0.1261 on `data/domain_manifest.jsonl`, above the 5% headroom threshold.
- **Heldout headroom still limited**: baseline WER stays ~0.0449 on `data/heldout_manifest.jsonl`.
- **Adapter efficacy confirmed**: LoRA weight delta was 592.7959 and interval WER tracked during headroom runs.

### Completed follow-ups
- [x] Expand training manifest to reach >= 200 samples after filtering (`data/train_manifest_expanded.jsonl`).
- [x] Build domain manifest with >= 200 samples (`data/domain_manifest.jsonl`).
- [x] Re-run baseline WER on the expanded domain manifest (`outputs/experiment_headroom_domain_20260218`).
