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
