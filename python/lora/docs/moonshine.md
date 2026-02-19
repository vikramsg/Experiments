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

### 4.1 Targeting the Decoder for "Inference Intelligence"
The parameter distribution (Table 1) dictates that LoRA should prioritize the **Decoder (22.8M parameters)** over the Encoder (7.39M parameters). 
- **The Rationale:** In domain adaptation, the model often "hears" the audio correctly via the encoder but fails to map those sounds to the correct domain-specific vocabulary. 
- **The Fix:** Focus LoRA adapters on the **Cross-Attention layers** of the decoder. These are the specific "bridges" where the model queries the encoder's acoustic features to generate text tokens. This is the most likely area to solve the "Loss ↓ but WER ↔" disconnect.

### 4.2 The Adapter Stage: The High-Leverage Target
The **1.31M parameter Adapter** is a specialized bridge that converts position-free features into position-aware ones.
- **The Rationale:** Domain-shifted audio (different accents, background noise) often disrupts the temporal alignment of speech. 
- **The Fix:** By including the Adapter stage in `target_modules`, we allow the model to specifically re-calibrate how it aligns sliding-window acoustic features with the decoder's token generation loop.

### 4.3 Preprocessing Parity (asinh & CMVN)
Moonshine v2 uses a specific sequence of **CMVN** and **asinh** nonlinearity in its preprocessor.
- **The Rationale:** Standard linear or Gaussian normalization used in many training scripts creates a "feature mismatch." The model then spends its limited LoRA capacity trying to "un-distort" the input rather than learning the domain language.
- **The Fix:** Ensure training data is fed through a preprocessor that matches the `asinh` scaling. Do not apply generic Min-Max normalization that could saturate the logarithmic range Moonshine expects.

### 4.4 Sliding-Window Stability (Long-and-Slow)
The use of local (sliding-window) attention means the model is less "globally aware" than Whisper.
- **The Rationale:** Gradients in sliding-window architectures can be noisier than full-attention models. Short training runs (e.g., 200 steps) are often insufficient for the weights to stabilize across these windows.
- **The Fix:** Increase training steps to **1000+** (at least 3-5 full epochs) with a **linear learning rate decay**. This gives the local attention windows enough "exposure" to the domain data to settle into a new global transcription logic.

## 5. Resources & Direct Links

- **Research Paper (Feb 2026):** [Moonshine v2: Ergodic Streaming Encoder ASR for Latency-Critical Speech Applications](https://arxiv.org/abs/2602.12241)
- **Official GitHub:** [usefulsensors/moonshine](https://github.com/usefulsensors/moonshine)
- **Hugging Face Models:** [UsefulSensors Moonshine v2 Collection](https://huggingface.co/UsefulSensors)
