# Plan: Fine-tune Moonshine Medium on Apple M3 (16GB)

## Context & Constraints
- **Hardware:** Apple M3 (8-core CPU, 10-core GPU, 16GB RAM). Use Metal/MPS when available; expect small batch sizes and aggressive memory saving.
- **Moonshine models:** README notes safetensors checkpoints are available via the Moonshine/UsefulSensors HF org; these are float checkpoints exported from the training pipeline.
- **PEFT:** Use LoRA adapters to keep trainable params small and fit on-device.

## Phase 0 — Environment + Feasibility Check
1. **Create a clean venv** and install: `torch` (MPS), `transformers`, `datasets`, `accelerate`, `peft`, `soundfile`, `evaluate`, `jiwer`.
2. **Verify MPS availability** with a short Python script (`torch.backends.mps.is_available()`), and record CPU fallback performance.
3. **Confirm Moonshine checkpoint format**: identify the **medium** model checkpoint and tokenizer/feature-extractor assets (safetensors preferred). If only ONNX is available, note conversion requirements.

## POC — Quick Proof of Concept (1–2 hours)
Goal: demonstrate a full LoRA fine-tune + inference loop on this Mac with minimal data.
1. **Lock Python/runtime**: use Python 3.11/3.12; create venv via `uv venv`, install deps via `uv sync`.
2. **Smoke test MPS**: run a tiny forward/backward pass on `mps` and confirm no fallback.
3. **Pick smallest viable ASR model**: use the smallest Moonshine-compatible checkpoint; otherwise a tiny HF ASR model supported by `transformers` + `peft`.
4. **Prepare tiny dataset**: 20–100 labeled utterances (5–10 minutes total), short clips (5–15s), WAV/FLAC + transcripts.
5. **Run LoRA fine-tune**:
   - `batch_size=1`
   - `gradient_accumulation_steps=8-16`
   - `max_steps=50-200`
   - `lora_r=4-8`
   - `learning_rate=1e-4 to 5e-4`
6. **Validate success**: loss decreases and a before/after transcript comparison shows improvement.
7. **Save + reload adapter**: verify inference uses the saved LoRA checkpoint.
8. **Record outcomes**: wall-clock time, peak memory, and any MPS issues.
9. **POC stopping criteria**: loss decreases, transcript improves, and saved adapter reloads for inference.

## Guidance
- I will not stop until the POC is complete, including running verification and confirming it behaves as intended.
- I will not stop for clarifications and will instead use best practices.

## Phase 1 — Tiny-Model Proof of Life (Required)
Goal: demonstrate the fine-tuning pipeline works end-to-end on this Mac before scaling up.
1. **Select a tiny ASR model** compatible with `transformers` + `peft` (smallest available Moonshine checkpoint if compatible; otherwise a tiny HF ASR model).
2. **Prepare a 5–10 minute toy dataset** (short WAV/FLAC clips + transcripts) to keep steps fast.
3. **Run LoRA fine-tuning** with tiny batch size (1), gradient accumulation, and short max steps (e.g., 100–300) on MPS.
4. **Validate success** by generating a transcript before/after and verifying loss decrease.

## Phase 2 — Moonshine Medium Checkpoint Prep
1. **Identify the exact medium model assets** (weights + tokenizer/feature extractor). Confirm parameter count (~245M from README).
2. **Load in PyTorch/Transformers**: confirm architecture is supported; if not, outline custom model wrapper.
3. **Define LoRA targets**: map attention and/or projection layers used in the Moonshine encoder/decoder.

## Phase 3 — Medium LoRA Fine-tune on M3
1. **Memory strategy:**
   - Use MPS + fp16/bf16 (whichever is stable).
   - Enable gradient checkpointing.
   - Batch size 1–2, gradient accumulation 8–16.
2. **Training loop:**
   - Use `accelerate` for device handling.
   - Log loss, WER on a small validation split.
3. **Runtime sanity checks:** monitor MPS memory, fallback to CPU if unstable.

## Phase 4 — Evaluation + Export
1. **Evaluate WER** on a small held-out set and compare to baseline.
2. **Save LoRA adapter** and document loading instructions.
3. **Optional:** merge adapter into base model and re-export (safetensors) if downstream tooling requires a single checkpoint.

## Deliverables
- Tiny-model training notebook/script (phase 1).
- Medium-model training script with LoRA config (phase 3).
- Evaluation report + adapter checkpoint (phase 4).

## Ref

- peft repo is at /tmp/peft
- moonshine repo is at /tmp/moonshine
