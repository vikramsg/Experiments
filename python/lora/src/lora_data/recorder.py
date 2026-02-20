"""Interactive audio recording tooling for dataset generation."""
import argparse
import json
import logging
import random
import sys
import time
import uuid
import wave
from pathlib import Path

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Attempt to import audio libs, handle graceful failure for CI/testing
try:
    import sounddevice as sd
    from pynput import keyboard
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logger.warning("sounddevice or pynput not installed. Interactive recording disabled.")

class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames = []
        self._stream = None

    def start(self):
        """Start capturing audio."""
        self.frames = []
        if not AUDIO_AVAILABLE:
            return
            
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            self.frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop capturing and return the audio data."""
        if not self._stream:
            return np.array([], dtype=np.int16)
            
        self._stream.stop()
        self._stream.close()
        self._stream = None
        
        if not self.frames:
            return np.array([], dtype=np.int16)
        return np.concatenate(self.frames, axis=0)

def save_wav(filename: str | Path, audio_data: np.ndarray, sample_rate: int = 16000):
    """Save raw int16 numpy array to a WAV file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

def append_to_manifest(manifest_path: str | Path, audio_path: str | Path, text: str):
    """Append a single entry to the JSONL manifest."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {"audio": str(audio_path), "text": text}
    with open(path, "a") as f:
        # Avoid literal backslash n bug entirely using Python's native newline
        f.write(json.dumps(entry))
        f.write("\n")

def run_interactive_session(prompts_file: str, out_dir: str, manifest_file: str):
    """Run the main interactive recording loop."""
    if not AUDIO_AVAILABLE:
        logger.error("Cannot run interactive session without sounddevice/pynput.")
        sys.exit(1)
        
    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
        
    if not prompts:
        logger.error(f"No prompts found in {prompts_file}")
        sys.exit(1)

    out_path = Path(out_dir)
    recorder = AudioRecorder()
    
    logger.info("=" * 50)
    logger.info("🎙️  PERSONALIZED VOICE RECORDER STARTED")
    logger.info("=" * 50)
    logger.info("Instructions:")
    logger.info("1. A prompt will appear on screen.")
    logger.info("2. Press and HOLD the SPACEBAR to record.")
    logger.info("3. Read the prompt out loud.")
    logger.info("4. Release the SPACEBAR to stop recording.")
    logger.info("5. The file is automatically saved.")
    logger.info("Press 'q' at any time to quit.")
    logger.info("-" * 50)

    is_recording = False
    current_prompt = ""
    shutdown_requested = False
    
    def pick_prompt():
        return random.choice(prompts)
        
    current_prompt = pick_prompt()
    logger.info(f"\n[PROMPT]: {current_prompt}")

    def on_press(key):
        nonlocal is_recording, shutdown_requested
        
        try:
            if key.char == 'q':
                shutdown_requested = True
                return False
        except AttributeError:
            pass

        if key == keyboard.Key.space and not is_recording:
            is_recording = True
            logger.info("🔴 RECORDING...")
            recorder.start()

    def on_release(key):
        nonlocal is_recording, current_prompt
        
        if key == keyboard.Key.space and is_recording:
            is_recording = False
            audio_data = recorder.stop()
            logger.info("⏹️  STOPPED")
            
            if len(audio_data) > 0:
                clip_id = str(uuid.uuid4())[:8]
                filename = out_path / f"clip_{clip_id}.wav"
                
                save_wav(filename, audio_data)
                append_to_manifest(manifest_file, filename, current_prompt)
                logger.info(f"✅ Saved to {filename}")
            else:
                logger.warning("⚠️  No audio captured. Try holding spacebar longer.")
            
            # Show next
            current_prompt = pick_prompt()
            logger.info(f"\n[PROMPT]: {current_prompt}")

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while not shutdown_requested:
            time.sleep(0.1)
            if not listener.running:
                break

    logger.info("\nSession ended.")

def run_headless_verification(out_dir: str, manifest_file: str):
    """Run an automated verification pass without human input."""
    logger.info("Running Headless Verification...")
    out_path = Path(out_dir)
    
    # Generate 1 second of 16kHz sine wave
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate, False)
    # 440 Hz sine wave, max amplitude 10000 (well within 16-bit range)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    
    # Ensure shape is (N, 1) for save_wav expectations if reshaping occurs, 
    # though 1D array is handled safely by tobytes()
    audio_data = audio_data.reshape(-1, 1)

    clip_id = "test_" + str(uuid.uuid4())[:8]
    filename = out_path / f"{clip_id}.wav"
    text = "this is a headless verification test"
    
    save_wav(filename, audio_data)
    append_to_manifest(manifest_file, filename, text)
    
    logger.info(f"✅ Headless verification complete. Generated: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Recorder Tooling")
    parser.add_argument(
        "--prompts", default="data/coding_prompts.txt", help="Path to prompts text file"
    )
    parser.add_argument(
        "--out-dir", default="data/raw_audio/my_voice", help="Directory to save WAV files"
    )
    parser.add_argument(
        "--manifest", default="data/my_voice_all.jsonl", help="Path to master JSONL manifest"
    )
    parser.add_argument(
        "--non-interactive", action="store_true", help="Run headless verification mode"
    )
    
    args = parser.parse_args()
    
    if args.non_interactive:
        run_headless_verification(args.out_dir, args.manifest)
    else:
        run_interactive_session(args.prompts, args.out_dir, args.manifest)