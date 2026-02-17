# Plan: Demonstrate Improved STT

## Outcome
- Demonstrate measurable STT improvement on real audio (WER improvement over baseline).

## Constraints + Assumptions
- Use `UsefulSensors/moonshine-tiny` on MPS.
- Use LibriSpeech real audio and a larger held-out manifest.
- Run long enough to observe WER movement (not just loss).
- FFmpeg is available for manifest decoding.

## What Was Wrong Last Time
- WER did not improve (baseline and tuned WER were both 1.0).
- Training was too small/short to move WER on real audio.

## Plan
1. **Dataset scope:** 800–1200 samples from `librispeech_asr` `train.100` with speaker split.
2. **Held-out manifest:** build 100–200 real audio clips from `validation` or `test` split.
3. **Training config:** `max_steps=1500–2500`, `batch_size=1`, `grad_accum=8–16`, `lr=2e-4–4e-4`.
4. **Baseline eval:** compute WER on the full held-out manifest before training.
5. **Fine-tune run:** train LoRA adapter, log loss over time.
6. **Post-train eval:** compute WER on the same held-out manifest.
7. **Artifacts + report:** save adapter, processor, metrics JSON, artifact test JSON, and update `docs/training_report.md`.

## Final Artifact Test (STT)
- Build a real held-out manifest:
  `uv run python scripts/build_manifest.py --split test --samples 120 --output data/heldout_manifest.jsonl`
- Run baseline STT (no adapter):
  `uv run python scripts/run_stt.py --model-id UsefulSensors/moonshine-tiny --processor-dir outputs/<run>/processor --audio-list data/heldout_manifest.jsonl --output outputs/<run>/baseline_stt.json --device mps`
- Run tuned STT (adapter):
  `uv run python scripts/run_stt.py --model-id UsefulSensors/moonshine-tiny --adapter-dir outputs/<run>/lora_adapter --processor-dir outputs/<run>/processor --audio-list data/heldout_manifest.jsonl --output outputs/<run>/tuned_stt.json --device mps`

## Verification
- Held-out manifest contains real audio arrays (not synthetic silence).
- `real_metrics.json` reports baseline + tuned WER and loss.
- `artifact_test.json` includes ≥100 samples and WER.

## Acceptance Criteria
- WER improves by ≥5% relative on the held-out manifest.
- Training completes without MPS fallback or crashes.

## Guidance
- I will not stop until the outcome is demonstrated.
- I will not stop for clarifications; I will make decisions using best practices and documentation.

## Ref

- peft repo is at /tmp/peft
- moonshine repo is at /tmp/moonshine
