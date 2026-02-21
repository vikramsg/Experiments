"""Interactive audio recording tooling for dataset generation."""
import argparse
import json
import logging
import random
import sys
import termios
import time
import tomllib
import uuid
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from pynput import keyboard
from rich.console import Console
from rich.panel import Panel

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()

class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames = []
        self._stream = None

    def start(self):
        """Start capturing audio."""
        self.frames = []
            
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

def parse_prompts(content: str) -> list[str]:
    """Parse TOML content into a list of prompts."""
    data = tomllib.loads(content)
    prompts = data.get("prompts", [])
    if not isinstance(prompts, list):
        raise ValueError("The 'prompts' key must be a list of strings")
    return [str(p).strip() for p in prompts if str(p).strip()]

def run_interactive_session(prompts_file: str, out_dir: str, manifest_file: str):
    """Run the main interactive recording loop."""
    with open(prompts_file) as f:
        prompts = parse_prompts(f.read())
        
    if not prompts:
        logger.error(f"No prompts found in {prompts_file}")
        sys.exit(1)

    out_path = Path(out_dir)
    recorder = AudioRecorder()
    
    console.print(Panel(
        f"[bold cyan]Instructions:[/bold cyan]\n"
        f"1. A prompt will appear on screen.\n"
        f"2. Press and [bold yellow]HOLD[/bold yellow] the "
        f"[bold green]SPACEBAR[/bold green] to record.\n"
        f"3. Read the prompt out loud.\n"
        f"4. Release the [bold green]SPACEBAR[/bold green] to stop recording.\n"
        f"5. The file is automatically saved.\n\n"
        f"[bold cyan]Storage:[/bold cyan]\n"
        f"• Audio Dir: [green]{out_path}[/green]\n"
        f"• Manifest:  [green]{manifest_file}[/green]\n\n"
        f"[dim]Press 'q' at any time to quit.[/dim]",
        title="🎙️  [bold magenta]PERSONALIZED VOICE RECORDER STARTED[/bold magenta]",
        border_style="cyan",
        expand=False
    ))

    is_recording = False
    current_prompt = ""
    shutdown_requested = False
    
    random.shuffle(prompts)
    prompt_iter = iter(prompts)
    
    def pick_prompt():
        try:
            return next(prompt_iter)
        except StopIteration:
            console.print("[bold yellow]No more prompts left in the file![/bold yellow]")
            return None
        
    current_prompt = pick_prompt()
    if not current_prompt:
        return
    
    console.print(
        f"\n[bold green]\\[PROMPT]:[/bold green] "
        f"[bold white]{current_prompt}[/bold white]"
    )

    def on_press(key):
        nonlocal is_recording, shutdown_requested
        
        if hasattr(key, 'char') and key.char == 'q':
            shutdown_requested = True
            return False

        if key == keyboard.Key.space and not is_recording:
            is_recording = True
            console.print("[bold red]🔴 RECORDING...[/bold red]")
            recorder.start()

    def on_release(key):
        nonlocal is_recording, current_prompt, shutdown_requested
        
        if key == keyboard.Key.space and is_recording:
            is_recording = False
            audio_data = recorder.stop()
            console.print("[bold bright_black]⏹️  STOPPED[/bold bright_black]")
            
            if len(audio_data) > 0:
                clip_id = str(uuid.uuid4())[:8]
                filename = out_path / f"clip_{clip_id}.wav"
                
                save_wav(filename, audio_data)
                append_to_manifest(manifest_file, filename, current_prompt)
                console.print(f"[bold green]✅ Saved to[/bold green] [cyan]{filename}[/cyan]")
            else:
                console.print(
                    "[bold yellow]⚠️  No audio captured. "
                    "Try holding spacebar longer.[/bold yellow]"
                )
            
            # Show next
            current_prompt = pick_prompt()
            if not current_prompt:
                shutdown_requested = True
                return False
                
            console.print(
                f"\n[bold green]\\[PROMPT]:[/bold green] "
                f"[bold white]{current_prompt}[/bold white]"
            )

    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    new_attr = termios.tcgetattr(fd)
    new_attr[3] = new_attr[3] & ~termios.ECHO  # Disable ECHO
    
    try:
        termios.tcsetattr(fd, termios.TCSANOW, new_attr)
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            while not shutdown_requested:
                time.sleep(0.1)
                if not listener.running:
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old_attr)

    console.print("\n[bold magenta]Session ended.[/bold magenta]")

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
        "--prompts", default="data/coding_prompts.toml", help="Path to prompts TOML file"
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