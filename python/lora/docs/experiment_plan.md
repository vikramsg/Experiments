# Experiment Plan

Central tracker for hypotheses, configs, and evaluation rules. Keep this in sync
with `docs/training.md` and reference run artifacts from `docs/training_report.md`.

## Evaluation Protocol
- Use the same manifest for baseline and tuned comparisons.
- Ensure inference preprocessing matches training preprocessing (RMS normalization).
- Use identical Moonshine decode settings across baseline, tuned, and in-training evals.
- Prefer full-manifest WER; avoid tiny sample sets unless explicitly marked POC.

## Active Hypotheses
| ID | Hypothesis | Dataset / Split | Config | Expected Outcome | Status |
| --- | --- | --- | --- | --- | --- |
| H1 | Use adapter-aware `generate` in evaluation to ensure active LoRA is respected. | LibriSpeech clean heldout | Eval path uses `PeftModel.generate` | WER parity between in-training eval and `scripts/run_stt.py`. | TODO |
| H2 | Increase training volume to reduce underfitting on 1k samples. | LibriSpeech clean train.100 | 2k+ steps, grad_accum=4 | Tuned WER improves or overfits predictably. | TODO |
| H3 | Revisit learning rate now that normalization is fixed. | LibriSpeech clean train.100 | 1e-4 to 3e-4 | Faster convergence without WER regression. | TODO |
| H4 | Evaluate on harder or domain-shifted data to measure real gains. | Noisy / accented / domain set | Same decode + normalization | Tuned WER improves on target domain. | TODO |

## Test Matrix
| Run ID | Command | Dataset | Steps | LR | LoRA Target Modules | Manifest | Baseline WER | Tuned WER | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |

## Run Notes
- Record any non-standard actions, fallbacks, or manual changes here with a date.
