# Moonshine v2 Technical Reference

**Source:** Kudlur et al., "Moonshine v2: Ergodic Streaming Encoder ASR for Latency-Critical Speech Applications", arXiv:2602.12241 (Feb 2026).

Moonshine v2 is a family of ergodic streaming-encoder ASR models designed for low-latency edge deployment. It replaces the full-attention encoder of v1 with sliding-window self-attention.

## 1. Architecture Breakdown (Table 1)

Moonshine v2 is divided into four distinct stages. The parameter distribution is heavily weighted toward the decoder.

| Model | Encoder Dim | Decoder Dim | Layers (E/D) | Pre (M) | Enc (M) | Adap (M) | Dec (M) | Total (M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tiny** | 320 | 320 | 6 / 6 | 2.08 | 7.39 | 1.31 | 22.80 | 33.57 |
| **Small** | 620 | 512 | 10 / 10 | 7.74 | 43.49 | 2.86 | 69.27 | 123.36 |
| **Medium**| 768 | 640 | 14 / 14 | 11.86 | 93.66 | 3.64 | 135.77 | 244.93 |

## 2. Component Details

### 2.1 Audio Preprocessor (Section 3.1)
- **Feature Rate:** 50 Hz (Standardized to align with Whisper).
- **Processing:** Audio is segmented into non-overlapping 80-sample windows (5ms at 16kHz).
- **Normalizations:** Per-frame Cepstral Mean and Variance Normalization (CMVN).
- **Nonlinearity:** **asinh** (inverse hyperbolic sine), chosen for logarithmic compression without saturation, balancing dynamic range.
- **Convolutions:** Two causal stride-2 convolutions reduce the frame rate to ~50 fps.

### 2.2 Streaming Encoder (Section 3.2)
- **Attention Type:** Sliding-window self-attention.
- **Window Configuration:**
  - First 2 and Last 2 layers: **(16, 4)** window (16 frames left context, 4 frames right/lookahead).
  - Intermediate layers: **(16, 0)** window (strictly causal).
- **Positional Embeddings:** **None (Position-free).** Computations are translation-invariant; the model infers structure from local context.
- **Latency:** 4 frames of lookahead results in an algorithmic latency of **80 ms** (4 * 20ms).

### 2.3 Adapter (Section 3.3)
- **Function:** Bridges the position-free encoder with the position-aware decoder.
- **Mechanism:** Adds **learned positional embeddings** to the encoder outputs and applies linear projection to match decoder dimensions.

### 2.4 Decoder (Section 3.4)
- **Type:** Standard causal Transformer.
- **Positionality:** Uses **Rotary Positional Embeddings (RoPE)**.
- **Complexity:** Uses **SwiGLU** feed-forward blocks (the encoder does not).

## 3. Benchmarks & Performance

### Word Error Rate (Table 3)
| Dataset | Tiny (34M) | Small (123M) | Medium (245M) |
| :--- | :---: | :---: | :---: |
| LibriSpeech (clean) | 4.49 | 2.49 | 2.08 |
| **LibriSpeech (other)** | **12.09** | **6.78** | **5.00** |
| GigaSpeech | 13.90 | 10.41 | 9.46 |
| **Average** | **12.01** | **7.84** | **6.65** |

### Latency & Load (Table 2 - Apple M3)
| Model | Latency (ms) | Compute Load (%) |
| :--- | :---: | :---: |
| **Moonshine v2 Tiny** | **50** | 8.03 |
| Whisper Tiny | 289 | 8.46 |
| Whisper Large v3 | 11286 | 330.65 |

## 4. LoRA Strategy Implications

- **Targeting:** Since the decoder (22.8M) is ~3x larger than the encoder (7.39M) and contains the cross-attention logic, LoRA adapters should prioritize **decoder projection layers**.
- **Adapter Fine-tuning:** The 1.31M parameter **Adapter** stage is a high-leverage target for domain-shifting, as it handles the transition from position-free acoustic features to token-aware embeddings.
