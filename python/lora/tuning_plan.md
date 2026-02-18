# Plan: LoRA Tuning Diagnostics (Data + Headroom)

## Goal
Identify why WER improvements are flat/regressing by validating data volume,
evaluation headroom, and adapter effectiveness before the next tuning sweep.

## Reference

docs/experiment_plan.md

## Ground Rules
1. **Use the unified experiment runner** (`src/lora/runners/experiment.py`) for any
   follow-up runs.
2. **Keep evaluation parity**: identical manifest, normalization, and decode
   settings across baseline and tuned runs.
3. **Do not modify legacy scripts** or bypass the unified runner.

## Hypotheses
1. **Data Volume (D1)**: Too few samples survive the duration filter, causing
   overfitting and flat WER.
2. **Headroom (D2)**: Baseline WER is already saturated on current manifests.
3. **Adapter Effect (D3)**: Adapters are not materially affecting inference after
   filtering and splits.

## Implementation Plan

### 1. Profile Manifest Durations
- **Inputs**: `data/train_manifest.jsonl`, `data/heldout_manifest.jsonl`,
  `data/test_manifest.jsonl`.
- **Action**: Count samples below 5s/8s/10s/15s/20s to pick a viable
  `--max-seconds` threshold.
- **Output**: Table with counts per threshold.

### 2. Validate Training Split Size
- **Config**: Use the chosen `--max-seconds` and compute train/val sizes.
- **Goal**: Ensure >= 200 samples in training before the next tuning sweep.

### 3. Measure Baseline Headroom
- **Config**: Run baseline WER with the chosen `--max-seconds` on
  heldout + domain manifests.
- **Goal**: Confirm baseline WER > 5% or add a harder domain manifest.

### 4. Adapter Efficacy Check
- **Config**: Short run (50–100 steps) on the filtered dataset.
- **Goal**: Verify LoRA weight deltas and WER direction matches the tuned run.

## Verification Criteria
1. **Data Sufficiency**: At least 200 training samples after filtering.
2. **Headroom**: Baseline WER > 5% on at least one manifest.
3. **Adapter Change**: Non-zero LoRA deltas and adapter-aware `generate` used.
4. **Parity**: Decode + normalization settings match between baseline and tuned.

## Acceptance Criteria
1. Chosen `--max-seconds` yields a stable train/val split with enough samples.
2. A manifest with measurable headroom is selected for future tuning.
3. Diagnostics summary recorded in `docs/experiment_plan.md` and `notes.md`.

## Completion Guidance (Do Not Stop Early)
1. **Finish all diagnostics before re-running H1–H4.**
2. If a diagnostic fails (e.g., low headroom), record it and adjust the manifest
   rather than changing hyperparameters.
3. **Always log exact commands, seeds, and output paths.**
4. **Run, debug, and complete every step end-to-end without stopping.**
5. Record any fallback device usage or manual adjustments in `notes.md` with the date and reason.
6. **Unblock yourself using best practices** when stuck

