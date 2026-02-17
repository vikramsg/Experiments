# Plan: Proper Training + Verification

## Goal
- Run a real tiny training pass with actual audio samples.
- Verify training artifacts and STT inference outputs are correct.
- Record results in `docs/training_report.md`.

## Known Issues From Last Run
- **Manifest audio decoding failed** without FFmpeg/torchcodec, so artifact test used synthetic silence.
- **Long runtimes** on larger sample sizes; quick run completed but is too small.
- **Split mismatch risk** if LibriSpeech split names are wrong (needs `train.100`).

## What Must Be Checked/Changed
- **FFmpeg availability confirmed** (2026-02-17) so manifest builder should use real audio.
- Use a real held-out manifest (no synthetic silence) for artifact STT.
- Verify split names and dataset streaming config are correct.
- Confirm MPS does not fall back to CPU during training.

## Plan
1. **Dataset prep:** use `librispeech_asr` streaming (`train.100`) with 200–500 samples and speaker-split.
2. **Manifest build (FFmpeg available):** generate a real held-out manifest with `scripts/build_manifest.py`.
3. **Training run:** execute `uv run python -m lora.runners.real_small` with `max_steps=500–1500`, `batch_size=1`, `grad_accum=8–16`.
4. **Baseline + tuned eval:** compute loss and WER on a fixed held-out split.
5. **Artifacts:** ensure adapter, processor, metrics JSON are saved under `outputs/<run>/`.
6. **Artifact STT test:** generate a real held-out manifest and run `uv run python -m lora.scripts.run_stt`.
7. **Report:** update `docs/training_report.md` with run config + results.

## Verification
- `outputs/<run>/real_metrics.json` exists with train/eval loss + WER.
- `outputs/<run>/artifact_test.json` exists with `samples` and `wer`.
- Adapter reloads and reproduces the artifact test output on rerun.
- Manifest uses real audio arrays (not synthetic silence).

## Acceptance Criteria
- Eval loss improves by ≥0.1 or WER improves by ≥5% relative.
- Training completes without crashes or MPS fallback.
- Artifact test runs on real audio (not synthetic silence).

## Guidance
- I will not stop until the entire plan is complete.
- I will not stop for clarifications; I will make decisions using best practices and documentation.

## Ref

- peft repo is at /tmp/peft
- moonshine repo is at /tmp/moonshine
