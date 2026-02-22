"""F5-TTS MLX engine encapsulation."""

from pathlib import Path

import librosa
import mlx.core as mx
import numpy as np
import soundfile as sf
from f5_tts_mlx.cfm import F5TTS
from f5_tts_mlx.utils import convert_char_to_pinyin

from lora_training.logging_utils import get_logger

LOGGER = get_logger(__name__)


class F5TTSEngine:
    """Wrapper around F5-TTS to enforce deterministic audio generation."""

    def __init__(self, ref_audio_path: str | Path, ref_audio_text: str):
        LOGGER.info("Loading F5TTS model...")
        self.f5tts = F5TTS.from_pretrained("lucasnewman/f5-tts-mlx")
        self.ref_audio_text = ref_audio_text
        self.hop_length = 256

        # Resample reference to 24kHz if it's 16kHz to prevent chipmunk effect
        audio, _ = librosa.load(str(ref_audio_path), sr=24000)
        audio = mx.array(audio)

        target_rms = 0.1
        rms = mx.sqrt(mx.mean(mx.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms

        self.ref_audio = audio

    def synthesize_audio(self, spoken_text: str, output_path: str | Path) -> float:
        """Generate audio from text using F5TTS and save to disk.

        Returns:
            Estimated duration of generated audio in seconds.
        """
        out_path = Path(output_path)
        if out_path.exists():
            return float(librosa.get_duration(path=str(out_path)))

        gen_text = convert_char_to_pinyin([self.ref_audio_text + " " + spoken_text])

        ref_audio_len = self.ref_audio.shape[0] // self.hop_length
        ref_text_len = len(self.ref_audio_text.encode("utf-8"))
        gen_text_len = len(spoken_text.encode("utf-8"))
        duration_in_frames = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)

        wave, _ = self.f5tts.sample(
            mx.expand_dims(self.ref_audio, axis=0),
            text=gen_text,
            duration=duration_in_frames,
            steps=16,
            method="rk4",
            speed=1.0,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
        )

        # Slice off the conditioned reference audio
        wave = wave[self.ref_audio.shape[0] :]
        mx.eval(wave)

        sf.write(str(out_path), np.array(wave), 24000)

        # Approximate duration
        return float(librosa.get_duration(path=str(out_path)))
