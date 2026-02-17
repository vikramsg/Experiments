# Fine-Tuning Plan for Moonshine Medium on Apple M3 (16GB)

## Phase 0: Feasibility and Environment Setup
- Confirm local hardware path: Apple M3 GPU (Metal/MPS), 16GB unified memory.
- Create a clean Python 3.11/3.12 environment (avoid Python 3.13 for compatibility risk).
- Install core stack: `torch`, `transformers`, `accelerate`, `datasets`, `evaluate`, `peft`, `jiwer`, `soundfile`, `librosa`.
- Validate MPS training path:
  - `torch.backends.mps.is_available()` is `True`.
  - One forward/backward pass runs on `mps`.
- Set conservative defaults for Apple Silicon:
  - `num_processes=1`
  - `dataloader_num_workers=0`
  - `PYTORCH_ENABLE_MPS_FALLBACK=1`
  - `fp16` where supported.

## Phase 1: Tiny Model Proof-of-Work (Must Happen First)
Goal: prove end-to-end fine-tuning works on this machine before medium.

- Use a tiny Moonshine-trainable checkpoint (or community Moonshine fine-tuning path) + PEFT LoRA.
- Build a tiny domain dataset (20-100 labeled utterances).
- Run a short LoRA fine-tune:
  - `batch_size=1`
  - `gradient_accumulation_steps=8-16`
  - `max_steps=50-200`
  - `lora_r=4-8`
  - `learning_rate=1e-4 to 5e-4`
- Success criteria:
  - training completes on MPS without OOM
  - training loss decreases
  - holdout WER improves vs baseline tiny
  - adapter checkpoint saves and reloads for inference
- Export/inference sanity check for the artifact path you intend to deploy.

## Phase 2: Medium Model Pilot on M3
Goal: safely move to Moonshine medium with memory-aware settings.

- Switch to medium trainable checkpoint.
- Start with strict memory settings:
  - LoRA on attention projections first (`q_proj`, `v_proj`), expand later if stable
  - `batch_size=1`
  - `gradient_accumulation_steps=16-64`
  - `gradient_checkpointing=True`
  - shorter clips first (5-15s), then increase
- Add periodic eval + checkpointing.
- Success criteria:
  - multi-hour stable run on MPS
  - no recurring OOM
  - measurable WER improvement on dev set

## Phase 3: Scale and Quality Optimization
Goal: make medium adapter robust for your domain.

- Increase dataset size and edge cases (noise, accents, jargon).
- Tune:
  - LoRA target modules
  - LoRA rank/alpha/dropout
  - LR schedule/warmup
  - sequence length and augmentation
- Run ablations and track WER/latency.
- Choose best checkpoint by quality and runtime tradeoff.

## Phase 4: Export, Quantize, and Benchmark for Deployment
Goal: produce production-ready artifacts for Moonshine runtime.

- Freeze best medium adapter.
- Convert/export into Moonshine runtime-compatible artifacts.
- Quantize with Moonshine tooling (where applicable) and validate parity.
- Benchmark on this exact M3 machine:
  - median and p95 latency
  - peak memory
  - WER on held-out domain set
- Package reproducible scripts for train/eval/export.

## Risks and Constraints
- 16GB unified memory is the primary limit; medium fine-tuning is feasible but tight.
- PEFT int8 example scripts that rely on bitsandbytes are not a direct drop-in for MPS.
- Moonshine runtime uses ONNX/ORT artifacts; training must target trainable checkpoint format first, then export.

## References
- https://github.com/moonshine-ai/moonshine
- https://github.com/huggingface/peft
- https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_adalora_whisper_large_training.py
- https://pytorch.org/docs/stable/notes/mps.html
- https://github.com/pierre-cheneau/moonshine-finetune

## Ref

- peft repo is at /tmp/peft
- moonshine repo is at /tmp/moonshine
