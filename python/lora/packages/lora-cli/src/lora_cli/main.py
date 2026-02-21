import time

import numpy as np
import typer
from pynput import keyboard
from rich.console import Console

from lora_cli.audio import AudioRecorder
from lora_cli.model import SpeechRecognizer
from lora_cli.ui import VoiceUI

app = typer.Typer()
console = Console()


@app.command()
def start(
    model_id: str = typer.Option("UsefulSensors/moonshine-tiny", help="Hugging Face model ID"),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", help="Run in simulation mode for testing"
    ),
):
    """Start the Moonshine Voice CLI."""
    ui = VoiceUI(console)
    recognizer = SpeechRecognizer(model_id=model_id, mock=non_interactive)

    if non_interactive:
        ui.print_system("Starting in NON-INTERACTIVE mode...")
        # Simulation Loop
        ui.update_status("Simulating: HOLDING SPACE")
        # Simulate recording (dummy audio)
        fake_audio = np.random.uniform(-0.5, 0.5, 16000 * 2).astype(
            np.float32
        )  # 2 seconds of noise
        time.sleep(1)

        ui.update_status("Simulating: RELEASED SPACE")
        ui.show_spinner("Transcribing...")
        text = recognizer.transcribe(fake_audio)
        ui.print_user_message(text)
        ui.print_system("Simulation Complete. Exiting.")
        return

    # Real Interactive Loop
    recorder = AudioRecorder()
    ui.print_system(f"Loaded model: {model_id}")
    ui.print_system("Ready. Hold SPACE to speak.")

    def on_press(key):
        if key == keyboard.Key.space:
            if not recorder.is_recording:
                recorder.start()
                ui.update_status("LISTENING", recording=True)

    def on_release(key):
        if key == keyboard.Key.space:
            if recorder.is_recording:
                audio_data = recorder.stop()
                ui.update_status("TRANSCRIBING", recording=False)
                ui.show_spinner("Thinking...")
                text = recognizer.transcribe(audio_data)
                ui.print_user_message(text)
                ui.reset_status()

    # Start Keyboard Listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    app()
