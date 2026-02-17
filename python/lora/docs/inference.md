# Inference Guide

## Overview

This guide explains how to run inference with your fine-tuned Moonshine models and how to optimize them for deployment on Apple Silicon.

## Current Inference Workflow (Python + PyTorch)

The current pipeline uses the `peft` library to dynamically load LoRA adapters on top of the base model at runtime.

### How it works
1.  **Load Base Model:** The original pre-trained model (e.g., `moonshine-tiny`) is loaded into memory (RAM/VRAM).
2.  **Load Adapters:** The small LoRA weights are loaded from your checkpoint directory.
3.  **Forward Pass:** The adapter weights are added to the base model's weights during computation.

### Usage
Run the standard inference script:
```bash
uv run python scripts/run_stt.py 
    --model-id UsefulSensors/moonshine-tiny 
    --adapter-dir outputs/real_small/lora_adapter 
    --processor-dir outputs/real_small/processor 
    --audio-list data/heldout_manifest.jsonl 
    --output outputs/results.json 
    --device mps
```

## Optimization Strategy (Apple Silicon)

To achieve maximum performance and efficiency on Mac (M-series chips), we recommend converting the model to **CoreML**. This unlocks the Apple Neural Engine (ANE), significantly reducing latency and power consumption.

### Step 1: Merge Adapters
First, you must "bake" the LoRA weights into the base model so it becomes a standard, standalone model.

**Why?**
- Eliminates the runtime overhead of calculating adapter additions.
- Simplifies the model architecture for conversion tools.

**Code Snippet:**
```python
from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq

# Load base + adapter
base_model = AutoModelForSpeechSeq2Seq.from_pretrained("UsefulSensors/moonshine-tiny")
model = PeftModel.from_pretrained(base_model, "path/to/adapter")

# Merge
model = model.merge_and_unload()

# Save standalone model
model.save_pretrained("outputs/merged_moonshine")
```

### Step 2: Convert to CoreML
Use `coremltools` to convert the merged PyTorch model into an Apple `.mlpackage`.

**Why?**
- **Neural Engine (ANE):** Runs on specialized hardware optimized for matrix math, freeing up your GPU and CPU.
- **Quantization:** Easily convert weights to `float16` (default) or `int8` to halve the model size with minimal accuracy loss.

**Code Snippet:**
```python
import coremltools as ct
import torch

# Trace the model (example)
example_input = torch.rand(1, 1, 16000) # Dummy audio
traced_model = torch.jit.trace(model, example_input)

# Convert
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],
    compute_precision=ct.precision.FLOAT16
)

mlmodel.save("MoonshineTiny.mlpackage")
```

### Step 3: Run with CoreML
You can run the `.mlpackage` using Python (for testing) or Swift (for native apps).

**Python Inference:**
```python
import coremltools as ct
import numpy as np

model = ct.models.MLModel("MoonshineTiny.mlpackage")
prediction = model.predict({"input_values": audio_data})
```

## Performance Comparison

| Feature | PyTorch (Current) | CoreML (Optimized) |
| :--- | :--- | :--- |
| **Compute Unit** | GPU (MPS) | Neural Engine (ANE) |
| **Model Format** | `safetensors` + `adapter.bin` | `.mlpackage` |
| **Precision** | FP32 / FP16 | FP16 / INT8 |
| **Latency** | ~50-100ms | ~10-20ms |
| **Power Usage** | High (GPU spikes) | Very Low (ANE) |

## Data Format
Optimization **does not change** the input data requirements.
- **Input:** Raw audio samples (16kHz, mono).
- **Preprocessing:** Normalization and tokenization remain identical.
- **Output:** Text tokens (same tokenizer).
