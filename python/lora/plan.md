# Plan: Small Real-World Test (Post-POC)

## Goal
- Validate the end-to-end fine-tuning pipeline on a realistic but small dataset.
- Produce a report with WER and loss comparisons against baseline.

## Plan
1. **Model selection:** keep `UsefulSensors/moonshine-tiny` for speed and MPS stability.
2. **Dataset scope:** sample 200–500 utterances (30–60 minutes total) from LibriSpeech (streaming).
3. **Split strategy:** 80/10/10 train/val/test with speaker separation to avoid leakage.
4. **Training config:** `batch_size=1`, `grad_accum=8–16`, `max_steps=500–1500`, `lr=2e-4–5e-4`, LoRA `r=8`, `alpha=16`, `dropout=0.05`.
5. **Baseline eval:** compute eval loss + WER on test split before training.
6. **Fine-tune run:** train LoRA adapter and log train/eval loss over time.
7. **Post-train eval:** compute eval loss + WER on the same test split.
8. **Artifacts:** save adapter, processor snapshot, metrics JSON, and update `docs/training_report.md`.

## Verification
- Adapter reloads and produces identical output on a fixed sample.
- Train and eval metrics are recorded for baseline and tuned runs.
- WER and loss are computed on the same fixed test split.

## Artifact Test (Speech-to-Text)
- Load the saved adapter + processor and run inference on 3–5 held-out audio clips.
- Compare transcripts against references and log WER.
- Save sample inputs, transcripts, and WER to `outputs/<run_name>/artifact_test.json`.
- **Script:** `scripts/run_stt.py`
- **Manifest script:** `scripts/build_manifest.py`
- **Manifest command:** `uv run python scripts/build_manifest.py --split test.clean --samples 5 --output data/heldout_manifest.jsonl`
- **STT command:** `uv run python scripts/run_stt.py --model-id UsefulSensors/moonshine-tiny --adapter-dir outputs/<run_name>/lora_adapter --processor-dir outputs/<run_name>/processor --audio-list data/heldout_manifest.jsonl --output outputs/<run_name>/artifact_test.json`
- **Args:** `--model-id`, `--adapter-dir`, `--processor-dir`, `--audio-list`, `--output`, `--device mps|cpu`

## Acceptance Criteria
- Eval loss decreases by ≥0.1 or WER improves by ≥5% relative to baseline.
- Training completes without MPS fallback or instability.
- Artifacts are written to `outputs/<run_name>/` with metrics JSON present.
- For smoke runs with synthetic/silence manifests, accept artifact test generation even if WER does not improve.

## Guidance
- I will not stop until the POC is complete, including running verification and confirming it behaves as intended.
- I will not stop for clarifications and will instead use best practices.

## Ref

- peft repo is at /tmp/peft
- moonshine repo is at /tmp/moonshine
