"""Interactive audio recording tooling for hard-negative mining and dataset correction."""

import argparse
import json
import logging
import sys
import termios
import time
import uuid
from pathlib import Path

import numpy as np
from pynput import keyboard
from rich.console import Console
from rich.panel import Panel

from lora_data.recorder import AudioRecorder, _append_to_manifest, _is_silent, save_wav
from lora_training.model_utils import (
    choose_device,
    configure_generation,
    load_processor,
)
from lora_training.transcribe import load_inference_model, run_moonshine_inference

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

console = Console()


def save_correction(
    audio_data: np.ndarray,
    corrected_text: str,
    out_dir: Path,
    manifest_path: Path,
    sample_rate: int = 16000,
) -> Path:
    """Save the audio data to a WAV file and append the corrected text to the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_id = str(uuid.uuid4())[:8]
    filename = out_dir / f"correction_{clip_id}.wav"

    save_wav(filename, audio_data, sample_rate=sample_rate)
    _append_to_manifest(manifest_path, filename, corrected_text)

    return filename


def run_interactive_session(args: argparse.Namespace):
    """Run the main interactive recording and transcription loop."""
    out_path = Path(args.out_dir)
    manifest_path = Path(args.manifest)

    device = choose_device(args.device)
    console.print(f"[bold cyan]Loading model:[/bold cyan] {args.model_id} on {device}")

    processor_path = args.processor_dir if args.processor_dir else args.model_id
    processor = load_processor(args.model_id, processor_path)

    model, config = load_inference_model(args.model_id, args.adapter_dir, device)
    configure_generation(model, processor)

    recorder = AudioRecorder()

    console.print(
        Panel(
            f"[bold cyan]Instructions:[/bold cyan]\n"
            f"1. Press and [bold yellow]HOLD[/bold yellow] "
            f"the [bold green]SPACEBAR[/bold green] to record.\n"
            f"2. Speak into your microphone.\n"
            f"3. Release the [bold green]SPACEBAR[/bold green] to stop recording and transcribe.\n"
            f"4. The model's transcription will be shown.\n"
            f"5. Answer if it is correct [y/n/q].\n"
            f"   - [green]y[/green]: Correct! Audio is discarded.\n"
            f"   - [red]n[/red]: Incorrect! You will type the correction, and it will be saved.\n"
            f"   - [yellow]q[/yellow]: Quit.\n\n"
            f"[bold cyan]Storage:[/bold cyan]\n"
            f"• Audio Dir: [green]{out_path}[/green]\n"
            f"• Manifest:  [green]{manifest_path}[/green]\n",
            title="🎙️  [bold magenta]CORRECTION RECORDER STARTED[/bold magenta]",
            border_style="cyan",
            expand=False,
        )
    )

    is_recording = False
    shutdown_requested = False

    def on_press(key):
        nonlocal is_recording, shutdown_requested

        if hasattr(key, "char") and key.char == "q":
            shutdown_requested = True
            return False

        if key == keyboard.Key.space and not is_recording:
            is_recording = True
            console.print("[bold red]🔴 RECORDING...[/bold red]")
            recorder.start()

    def on_release(key):
        nonlocal is_recording, shutdown_requested

        if key == keyboard.Key.space and is_recording:
            is_recording = False
            audio_data = recorder.stop()
            console.print("[bold bright_black]⏹️  STOPPED[/bold bright_black]")

            if _is_silent(audio_data, threshold=500):
                console.print(
                    "[bold yellow]⚠️  No sound detected (or too quiet). "
                    "Please try again.[/bold yellow]"
                )
                console.print(
                    "\n[bold cyan]Ready... Hold SPACEBAR to record, or 'q' to quit.[/bold cyan]"
                )
                return

            console.print("[bold cyan]Transcribing...[/bold cyan]")

            # Convert audio data to float list for inference
            # Audio is int16, need to normalize to [-1.0, 1.0] float for Moonshine processor
            # but normalize_audio_rms expects a list of floats
            audio_float = audio_data.flatten().astype(np.float32) / 32768.0

            prediction = run_moonshine_inference(model, processor, audio_float.tolist(), device)

            console.print(
                f"\n[bold magenta]Model heard:[/bold magenta] [bold white]{prediction}[/bold white]"
            )

            # We must restore terminal attributes to allow user input
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, old_attr)

            try:
                while True:
                    response = input("Is this correct? [y/n/q]: ").strip().lower()
                    if response in ("y", "n", "q"):
                        break
                    print("Invalid input. Please enter y, n, or q.")

                if response == "q":
                    shutdown_requested = True
                    return False
                elif response == "y":
                    console.print("[bold green]✅ Correct! Discarding audio.[/bold green]")
                elif response == "n":
                    corrected_text = input("Please type the CORRECT transcription: ").strip()
                    if corrected_text:
                        saved_path = save_correction(
                            audio_data, corrected_text, out_path, manifest_path
                        )
                        console.print(
                            f"[bold green]✅ Saved correction to[/bold green] "
                            f"[cyan]{saved_path}[/cyan]"
                        )
                    else:
                        console.print(
                            "[bold yellow]⚠️ Empty correction provided. "
                            "Discarding audio.[/bold yellow]"
                        )
            finally:
                # Disable ECHO again for the spacebar listener
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, new_attr)

            if not shutdown_requested:
                console.print(
                    "\n[bold cyan]Ready... Hold SPACEBAR to record, or 'q' to quit.[/bold cyan]"
                )

    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    new_attr = termios.tcgetattr(fd)
    new_attr[3] = new_attr[3] & ~termios.ECHO  # Disable ECHO

    try:
        termios.tcsetattr(fd, termios.TCSANOW, new_attr)
        console.print("\n[bold cyan]Ready... Hold SPACEBAR to record, or 'q' to quit.[/bold cyan]")
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            while not shutdown_requested:
                time.sleep(0.1)
                if not listener.running:
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old_attr)

    console.print("\n[bold magenta]Session ended.[/bold magenta]")


def run_headless_verification(args: argparse.Namespace):
    """Run an automated verification pass without human input."""
    logger.info("Running Headless Verification...")
    out_path = Path(args.out_dir)
    manifest_path = Path(args.manifest)

    device = choose_device(args.device)
    logger.info(f"Loading model: {args.model_id} on {device}")

    processor_path = args.processor_dir if args.processor_dir else args.model_id
    processor = load_processor(args.model_id, processor_path)

    model, config = load_inference_model(args.model_id, args.adapter_dir, device)
    configure_generation(model, processor)

    # Generate 1 second of 16kHz sine wave
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate, False)
    # 440 Hz sine wave, max amplitude 10000 (well within 16-bit range)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    audio_data = audio_data.reshape(-1, 1)

    logger.info("Transcribing fake audio...")
    audio_float = audio_data.flatten().astype(np.float32) / 32768.0
    prediction = run_moonshine_inference(model, processor, audio_float.tolist(), device)
    logger.info(f"Prediction: {prediction}")

    corrected_text = "this is a headless verification test correction"
    saved_path = save_correction(audio_data, corrected_text, out_path, manifest_path)

    logger.info(f"✅ Headless verification complete. Generated: {saved_path}")

    # Check manifest
    with open(manifest_path) as f:
        lines = f.readlines()
        last_line = json.loads(lines[-1])
        logger.info(f"Manifest last entry: {last_line}")
        assert last_line["text"] == corrected_text
        assert str(saved_path) in last_line["audio"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correction Recorder for Hard-Negative Mining")
    parser.add_argument("--model-id", default="UsefulSensors/moonshine-tiny", help="Base model ID")
    parser.add_argument("--adapter-dir", default=None, help="Path to LoRA adapter directory")
    parser.add_argument(
        "--processor-dir", default=None, help="Path to processor directory (optional)"
    )
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default=None)
    parser.add_argument(
        "--out-dir", default="data/raw_audio/corrections", help="Directory to save WAV files"
    )
    parser.add_argument(
        "--manifest", default="data/corrections_manifest.jsonl", help="Path to JSONL manifest"
    )
    parser.add_argument(
        "--non-interactive", action="store_true", help="Run headless verification mode"
    )

    args = parser.parse_args()

    if args.non_interactive:
        run_headless_verification(args)
    else:
        run_interactive_session(args)
