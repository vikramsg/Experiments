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

## Archive
- Legacy issues and incorrect setup outcomes are preserved in:
  - `docs/archive_legacy_results.md`
