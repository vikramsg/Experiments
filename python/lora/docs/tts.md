# F5-TTS Voice Cloning: Learnings & Best Practices

This document outlines the critical learnings, pitfalls, and best practices discovered while using `f5-tts-mlx` for synthetic data generation in this repository.

## 1. Sample Rate Mismatches (The "Chipmunk" Effect)
**Issue:** When raw 16kHz audio (standard for STT models like Moonshine) was passed as reference audio to F5-TTS, the resulting synthetic audio sounded like a high-pitched, fast-talking cartoon character.
**Root Cause:** F5-TTS MLX natively operates at and expects **24,000 Hz (24kHz)** audio. If 16kHz audio is passed as an array without resampling, the model misinterprets the sample rate, effectively treating the reference voice as being spoken 1.5x faster.
**Solution:** Always explicitly resample the reference audio to 24kHz before passing it to the model.
```python
import librosa
audio, sr = librosa.load(ref_audio_path, sr=24000)
```

## 2. Symbol Hallucinations & Phonetic Spelling
**Issue:** Technical jargon often contains symbols (e.g., `@refactor.md`, `justfile`, `/tmp`). Text-to-Speech models struggle to intuitively map raw symbols to phonetic sounds unless explicitly trained on them, resulting in skipped words, Chinese language fallbacks, or complete hallucinations.
**Solution:** Implement a two-track text system:
1.  **Spoken Text (TTS Input):** Convert symbols into their spoken English equivalents (e.g., `@` -> ` at `, `_` -> ` underscore `, `.md` -> ` dot m d `). Feed this *phonetic* string to the TTS model.
2.  **Target Text (STT Label):** Keep the raw symbols in the final `manifest.jsonl`.
This forces the downstream STT model to learn the mapping from the spoken sound "at" back to the symbol `@`.

## 3. Duration Calculation
**Issue:** F5-TTS MLX does not use seconds for the `duration` parameter; it uses **frames** based on a `HOP_LENGTH` of 256. Miscalculating this causes extreme generation timeouts or truncated audio.
**Solution:** Calculate duration proportionally to the reference audio length in frames, adjusted for the character length of the generated text versus the reference text.
```python
HOP_LENGTH = 256
ref_audio_len = audio.shape[0] // HOP_LENGTH
ref_text_len = len(ref_audio_text.encode('utf-8'))
gen_text_len = len(spoken_text.encode('utf-8'))
duration_in_frames = ref_audio_len + int((ref_audio_len / ref_text_len) * gen_text_len)
```

## 4. Inference Parameters
After testing, the following parameters provided the best balance of speed, stability, and voice similarity for Apple Silicon (MPS):
*   `steps=16`: Minimum required to avoid garbled speech.
*   `speed=1.0`: Natural pacing when sample rate is correctly 24kHz.
*   `cfg_strength=2.0`: Keeps the voice heavily conditioned on the reference audio.
*   `sway_sampling_coef=-1.0`: Default ODE solver stability factor.
*   `method="rk4"`: Standard Runge-Kutta solver.