# Known Issues

## 1. Heldout Metric Headroom Is Limited

**Status:** Open
**Date:** 2026-02-19
**Impact:** Small changes are hard to detect on heldout (`~0.045` baseline WER), so many runs appear flat.

### Current Guidance
- Treat heldout as a regression guardrail.
- Use domain manifest (`data/domain_manifest.jsonl`) as the primary optimization target.

## 2. Final-Checkpoint Bias

**Status:** Open
**Date:** 2026-02-19
**Impact:** Best interval WER can occur before the final step, but current workflow saves final adapter only.

### Current Guidance
- Track interval WER during training.
- Add best-checkpoint saving and/or patience-based early stopping in the next runner update.

## 3. Evaluation Sample Heterogeneity

**Status:** Open
**Date:** 2026-02-19
**Impact:** The `domain_manifest.jsonl` (200 samples) exhibits a sharp difficulty imbalance. WER typically hovers around 0.14-0.16 for the first 150 samples but "dives" to ~0.11 in the final 50 samples.

This indicates that the final samples in the manifest are objectively easier for the model. Because the WER hasn't leveled off by batch 200, the reported final metric is highly sensitive to the manifest order and may represent a "lucky" average rather than a stable performance indicator.

### Current Guidance
- Do not rely on small WER fluctuations (< 0.01) at the end of a 200-sample run.
- When generating new manifests via `scripts/build_domain_manifest.py`, consider increasing the sample count to 500 or higher to reach a stable convergence point.
- Monitor the step-by-step evaluation logs to ensure the metric has stabilized before concluding an experiment.

## Archive
- Legacy issues and incorrect setup outcomes are preserved in:
  - `docs/archive_legacy_results.md`
