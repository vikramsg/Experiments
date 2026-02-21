# Legacy Results Archive

Archived on: 2026-02-19  
Purpose: preserve pre-refresh experiment results and failure modes while keeping active docs focused on the current workflow.

## Why This Was Archived
- Earlier runs mixed low-data, short-duration filtering (`--max-seconds=8`) with saturated heldout evaluation.
- This produced unstable or flat WER behavior and made iteration noisy.
- Active docs now focus on expanded-manifest (`--max-seconds=20`) experiments.

## Legacy Experiment Matrix (Pre-Refresh)

| Run ID | Date | Dataset | Max Seconds | Steps | LR | Manifest | Baseline WER | Tuned WER | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h1-20260218 | 2026-02-18 | `data/train_manifest.jsonl` | 8 | 200 | 1e-5 | `data/heldout_manifest.jsonl` | 0.0462 | 0.0462 | Flat WER, loss down. |
| h2-20260218 | 2026-02-18 | `data/train_manifest.jsonl` | 8 | 2000 | 1e-4 | `data/heldout_manifest.jsonl` | 0.0462 | 0.0582 | Regressed; early stop threshold hit. |
| h3-lr1e4-20260218 | 2026-02-18 | `data/train_manifest.jsonl` | 8 | 200 | 1e-4 | `data/heldout_manifest.jsonl` | 0.0462 | 0.0462 | Temporary interval wobble; final flat. |
| h3-lr3e4-20260218 | 2026-02-18 | `data/train_manifest.jsonl` | 8 | 200 | 3e-4 | `data/heldout_manifest.jsonl` | 0.0462 | 0.0701 | Regressed. |
| h4-20260218 | 2026-02-18 | `data/train_manifest.jsonl` | 8 | 200 | 1e-4 | `data/test_manifest.jsonl` | 0.0000 | 0.0000 | 2-sample eval manifest; no measurable headroom. |

## Legacy Diagnostics (Pre-Refresh)

### Duration profile under `--max-seconds=8`

| Manifest | Samples | Mean (s) | <=8s |
| --- | --- | --- | --- |
| `data/train_manifest.jsonl` | 120 | 13.33 | 8 (6.7%) |
| `data/heldout_manifest.jsonl` | 120 | 6.94 | 85 (70.8%) |
| `data/test_manifest.jsonl` | 2 | 6.87 | 2 (100.0%) |

### Split sizes under `--max-seconds=8` (seed=42)

| Manifest | Filtered | Train | Val | Split |
| --- | --- | --- | --- | --- |
| `data/train_manifest.jsonl` | 8 | 7 | 1 | random |
| `data/heldout_manifest.jsonl` | 85 | 68 | 17 | random |
| `data/test_manifest.jsonl` | 2 | 1 | 1 | random |

## Legacy Training Report Snapshot

Source archived from: `docs/training_report.md` (pre-refresh version)

- Model: `UsefulSensors/moonshine-tiny`
- Dataset: `data/train_plus_heldout_manifest.jsonl` (240 samples)
- Additional manifest: `data/train_manifest_short_10s.jsonl` (17 samples)
- Command family: `lora.runners.real_small` with low-LR, long-step runs
- Reproduced outcome:
  - Heldout WER: `0.04458 -> 0.04417` (tiny gain)
  - Short-manifest WER: unchanged (`0.05085`)

## Legacy Failure Mode Summary

1. `--max-seconds=8` caused severe data starvation on `data/train_manifest.jsonl`.
2. Heldout benchmark headroom was too limited for stable visible gains.
3. Final-checkpoint-only saving could miss best interval WER during a run.
4. LoRA updates occurred, but gains were small and highly configuration-sensitive.
