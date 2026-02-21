import json
import os
import subprocess
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lora_training.logging_utils import setup_logging, get_logger

def spell_out_for_tts(text: str) -> str:
    replacements = {
        "@": " at ",
        "_": " underscore ",
        "/": " slash ",
        ".md": " dot m d ",
        ".jsonl": " dot json l ",
        ".out": " dot out ",
        "uv": " u v ",
        "WER": " w e r ",
        "f5-tts-mlx": " f 5 t t s m l x ",
        "tmp": " temp ",
        "nohup": " no hup ",
        "justfile": " just file "
    }
    spoken_text = text
    for symbol, spoken in replacements.items():
        spoken_text = spoken_text.replace(symbol, spoken)
    return " ".join(spoken_text.split())

def main():
    log_path = Path("/tmp/tts_verify/verify_final.log")
    setup_logging(log_path=log_path)
    logger = get_logger("tts_verification")

    logger.info("=== TTS Verification Script (Final Judgement Run) ===")
    
    out_dir = Path("/tmp/tts_verify")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ref_audio_path = "data/raw_audio/my_voice/clip_ec16024b.wav"
    ref_audio_text = "put a to do docstring at the top of the build domain manifest script about this issue. put in known issues document. Don't put a single or two liner, be a little more detailed."
    
    test_sentences = [
        "Please check the @refactor.md file.",
        "Did you run justfile yet?",
        "The WER is showing a high error rate.",
        "Update the /tmp/nohup.out with the new parameters.",
        "Why is f5-tts-mlx failing on the lora_adapter?"
    ]
    
    logger.info("Loading F5-TTS...")
    import mlx.core as mx
    from f5_tts_mlx.cfm import F5TTS
    from f5_tts_mlx.utils import convert_char_to_pinyin
    import soundfile as sf
    import librosa
    import numpy as np

    f5tts = F5TTS.from_pretrained("lucasnewman/f5-tts-mlx")
    
    logger.info("Loading and resampling reference audio to 24kHz...")
    audio, sr = librosa.load(ref_audio_path, sr=24000)
    audio = mx.array(audio)
    target_rms = 0.1
    rms = mx.sqrt(mx.mean(mx.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
        
    HOP_LENGTH = 256
    results = []
    
    for i, text in enumerate(test_sentences):
        logger.info("\n--- Generating Audio %d ---", i)
        spoken_text = spell_out_for_tts(text)
        logger.info("Target: %s", text)
        logger.info("Spoken: %s", spoken_text)
        
        gen_text = convert_char_to_pinyin([ref_audio_text + " " + spoken_text])
        
        ref_audio_len = audio.shape[0] // HOP_LENGTH
        ref_text_len = len(ref_audio_text.encode('utf-8'))
        gen_text_len = len(spoken_text.encode('utf-8'))
        duration_in_frames = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)
        
        logger.info("Generating (duration=%d frames, steps=16)...", duration_in_frames)
        wave, _ = f5tts.sample(
            mx.expand_dims(audio, axis=0),
            text=gen_text,
            duration=duration_in_frames,
            steps=16,
            method="rk4",
            speed=1.0,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0
        )
        
        wave = wave[audio.shape[0] :]
        mx.eval(wave)
        
        out_wav = out_dir / f"final_test_{i}.wav"
        sf.write(str(out_wav), np.array(wave), 24000)
        logger.info("Saved audio to: %s", out_wav)
        
        results.append({
            "target": text,
            "spoken": spoken_text,
            "audio_path": str(out_wav)
        })

    logger.info("\n=== FILES READY FOR USER JUDGEMENT ===")
    for r in results:
        logger.info("Audio:  %s", r['audio_path'])
        logger.info("Target: %s", r['target'])
        logger.info("---")

if __name__ == "__main__":
    main()
