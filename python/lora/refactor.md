# Refactor Notes

## How to Make a Real Tiny Small Run Work

The goal is a **real** tiny run with actual audio (not synthetic silence) while keeping runtime short.

### Why the Artifact Test Used a Placeholder Manifest
- The artifact test used a synthetic “silence” manifest because LibriSpeech decoding failed without FFmpeg/torchcodec.
- The training run still used real LibriSpeech audio via streaming + manual `soundfile` decode, but the held‑out manifest did not.
- Once FFmpeg/torchcodec (or an alternate decoder) is available, switch the manifest builder back to real audio.

### Recommended Minimal Run
- **Dataset:** `librispeech_asr` (streaming) using `train.100` split.
- **Samples:** 20–50 utterances.
- **Steps:** 20–50 max steps.
- **Batch size:** 1
- **Grad accumulation:** 4–8
- **WER batches:** 1–3

Example command:

```
uv run python train_real_small.py \
  --train-split train.100 \
  --dataset-samples 20 \
  --max-steps 20 \
  --gradient-accumulation-steps 4 \
  --wer-batches 1 \
  --output-dir outputs/real_small_quick
```

### Real Held‑Out Manifest Without FFmpeg Issues
Use `scripts/build_manifest.py` with FFmpeg installed to create a real held‑out manifest
from the `test` or `validation` splits.

### Debug Steps (FFmpeg Available)
1. Verify FFmpeg is installed and on PATH:
   `ffmpeg -version`
2. Verify torchcodec can load FFmpeg libraries:
   `uv run python -c "import torchcodec"`
3. Build a real manifest:
   `uv run python scripts/build_manifest.py --split test --samples 3 --output data/heldout_manifest.jsonl`
4. Confirm the manifest contains JSONL with `audio` arrays and `text` fields:
   `head -n 1 data/heldout_manifest.jsonl`

### Artifact Test With Real Audio
1. Generate manifest (real audio).
2. Run STT with saved adapter + processor.
3. Save outputs to `outputs/<run>/artifact_test.json`.

Commands:

```
uv run python scripts/build_manifest.py --split test --samples 3 --output data/heldout_manifest.jsonl
uv run python scripts/run_stt.py \
  --model-id UsefulSensors/moonshine-tiny \
  --adapter-dir outputs/real_small_quick/lora_adapter \
  --processor-dir outputs/real_small_quick/processor \
  --audio-list data/heldout_manifest.jsonl \
  --output outputs/real_small_quick/artifact_test.json
```

## Verification Criteria
- `train_real_small.py` completes with metrics JSON written to `outputs/<run>/real_metrics.json`.
- `scripts/build_manifest.py` emits JSONL rows with `audio` arrays and `text` values.
- `scripts/run_stt.py` produces `outputs/<run>/artifact_test.json` with `samples` and `wer`.

## Acceptance Criteria
- Training produces a non-zero `train_loss` and a valid `baseline_eval_loss`/`tuned_eval_loss`.
- Artifact test outputs a JSON report without errors.
- All refactor steps are applied and documented in `README.md` and `plan.md`.

## Refactor Plan: Scripts Layout

Move to a package-style layout to make imports clean and improve testability.

### Target Structure

```
src/lora/
  __init__.py
  data_loader.py
  evaluation.py
  model_utils.py
  training_config.py
  runners/
    poc.py
    real_small.py
  scripts/
    run_stt.py
    build_manifest.py
```

### Refactor Steps
1. Create `src/lora/` and move shared modules there.
2. Move `main.py` logic into `src/lora/runners/poc.py`.
3. Move `train_real_small.py` into `src/lora/runners/real_small.py`.
4. Update entrypoints to import from `src/lora`.
5. Add a thin `main.py` that delegates to `lora.runners.poc`.
6. Update `README.md` and `plan.md` command examples.

## Test Plan

Add `pytest` coverage for key behaviors with network/model downloads mocked.

### Tests
1. **Data loader:** `tests/test_data_loader.py`
   - `split_by_speaker` produces non-empty splits.
   - `prepare_dataset` yields expected tensor fields.
2. **Evaluation helpers:** `tests/test_evaluation.py`
   - `summarize_losses` returns correct mean and handles empty list.
3. **STT script:** `tests/test_run_stt.py`
   - Manifest parsing and normalization.
   - Error on empty manifest.

### Test Execution

```
make test
```

## Guidance
- I will not stop until the entire plan is complete.
- I will not stop for clarifications; I will make decisions using best practices and documentation.
