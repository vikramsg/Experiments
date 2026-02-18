# Known Issues

## 1. Word Error Rate (WER) Calculation

**Status:** Identified
**Date:** 2026-02-17
**Impact:** Evaluation metrics show WER of 1.0 (100% error) despite significant loss reduction.

### Description
The project currently uses the standard Hugging Face `evaluate` library's WER metric, which performs a strict case-sensitive comparison.

- **Reference Text (LibriSpeech):** Fully UPPERCASE (e.g., "HELLO WORLD").
- **Model Output (Moonshine):** Sentence case with punctuation (e.g., "Hello world.").

This mismatch causes every word to be counted as a substitution error, resulting in a WER of 1.0 even when the transcription is semantically correct.

### Reproduction
```python
from evaluate import load
wer = load("wer")
# Returns 1.0 because of case mismatch
wer.compute(predictions=["hello world"], references=["HELLO WORLD"])
```

### Proposed Solution
Implement text normalization in `src/lora/evaluation.py` before passing predictions and references to the metric:
1.  Convert both to lowercase.
2.  Remove punctuation.

## 2. WER Regression After Quick LoRA Run

**Status:** Identified
**Date:** 2026-02-18
**Impact:** Tuned WER is worse than baseline on both held-out and in-domain manifests.

### Description
Quick LoRA runs (e.g., 200 steps/200 samples) can regress WER even on in-domain data. The tuned adapter in
`outputs/real_quick_run_20260217c` increased WER from `0.0446` to `0.0638` on the held-out manifest and from
`0.0364` to `0.0521` on a train-domain manifest when re-evaluated with `scripts/run_stt.py`.

### Likely Contributors
- Training preprocessing uses raw audio while inference applies RMS normalization, causing distribution drift.
- Training run is too short to provide stable improvements.
- In-training evaluation uses different decode settings from the final STT script.

### Proposed Solution
- Apply RMS normalization in the training data pipeline to match inference.
- Increase training steps/samples per `plan.md`.
- Align evaluation decode settings with `scripts/run_stt.py` for Moonshine.
- If regressions persist, try DoRA (`use_dora=True`) or PiSSA initialization (`init_lora_weights="pissa"`).
