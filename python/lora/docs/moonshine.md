# Moonshine v2 Technical Reference

**Source:** Kudlur et al., "Moonshine v2: Ergodic Streaming Encoder ASR for Latency-Critical Speech Applications", arXiv:2602.12241 (Feb 2026).

Moonshine v2 is the second generation of speech recognition models from Moonshine AI. **The v2 architecture is currently hosted under the "Streaming" model IDs on Hugging Face** (e.g., `UsefulSensors/moonshine-streaming-tiny`).

## 1. Architecture Breakdown (Table 1)

Moonshine v2 is divided into four stages. The parameter distribution is heavily weighted toward the decoder.

| Model | Encoder Dim | Decoder Dim | Layers (E/D) | Pre (M) | Enc (M) | Adap (M) | Dec (M) | Total (M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tiny** | 320 | 320 | 6 / 6 | 2.08 | 7.39 | 1.31 | 22.80 | 33.57 |
| **Small** | 620 | 512 | 10 / 10 | 7.74 | 43.49 | 2.86 | 69.27 | 123.36 |
| **Medium**| 768 | 640 | 14 / 14 | 11.86 | 93.66 | 3.64 | 135.77 | 244.93 |

## 2. Component Details (Verified v2)

### 2.1 Audio Preprocessor
- **Feature Rate:** 50 Hz (Aligned with Whisper).
- **Processing:** Audio is segmented into 80-sample windows (5ms at 16kHz).
- **Normalizations:** Per-frame Cepstral Mean and Variance Normalization (CMVN).
- **Nonlinearity:** **asinh** (inverse hyperbolic sine) for logarithmic compression.

### 2.2 Streaming Encoder
- **Attention Type:** Sliding-window self-attention.
- **Window Configuration:**
  - First 2 and Last 2 layers: **(16, 4)** window (16 frames left context, 4 frames right/lookahead).
  - Intermediate layers: **(16, 0)** window (strictly causal).
- **Positional Embeddings:** **None (Position-free).** The model is translation-invariant.
- **Latency:** 4 frames of lookahead = **80 ms** algorithmic latency.

### 2.3 Adapter
- **Function:** Bridges the position-free encoder with the position-aware decoder.
- **Mechanism:** Adds learned positional embeddings and applies linear projection.

### 2.4 Decoder
- **Type:** Standard causal Transformer using **RoPE** and **SwiGLU** feed-forward blocks.

## 3. Benchmarks & Performance (v2 Tiny)

### Word Error Rate (Table 3)
| Dataset | WER (%) |
| :--- | :---: |
| LibriSpeech (clean) | 4.49 |
| **LibriSpeech (other)** | **12.09** |
| GigaSpeech | 13.90 |
| **Average** | **12.01** |

### Latency & Load (Table 2 - Apple M3)
| Model | Latency (ms) | Compute Load (%) |
| :--- | :---: | :---: |
| **Moonshine v2 Tiny** | **50** | 8.03 |
| Whisper Tiny | 289 | 8.46 |

## 4. Training & Fine-Tuning Strategy

### 4.1 The "Bridge-Focus" Approach
Given the parameter distribution, LoRA should prioritize the **Decoder (22.8M)** and the **Adapter (1.31M)**.
- **Rationale:** Most "linguistic intelligence" and the acoustic-to-text mapping (Cross-Attention) reside in these components. Fine-tuning them solves the "Loss ↓ but WER ↔" disconnect.

### 4.2 Preprocessing Parity
- **Sample Rate:** 16,000 Hz.
- **Target RMS:** **0.075**.
- **Normalization:** 32-bit float, range **[-1.0, 1.0]**.

## 5. Resources
- **Research Paper (Feb 2026):** [arXiv:2602.12241](https://arxiv.org/abs/2602.12241)
- **Official GitHub:** [moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine)
- **Hugging Face:** [UsefulSensors Moonshine Collection](https://huggingface.co/UsefulSensors)
