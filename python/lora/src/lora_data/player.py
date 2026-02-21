"""Interactive audio player for reviewing generated clips."""

import argparse
import logging
import sys
import termios
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf
from pynput import keyboard
from rich.console import Console
from rich.panel import Panel

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()


def run_interactive_player(audio_dir: str):
    """Run the main interactive playback loop."""
    dir_path = Path(audio_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        console.print(f"[bold red]Directory not found: {audio_dir}[/bold red]")
        sys.exit(1)

    audio_files = sorted(list(dir_path.glob("*.wav")))
    if not audio_files:
        console.print(f"[bold red]No .wav files found in {audio_dir}[/bold red]")
        sys.exit(1)

    total_files = len(audio_files)
    current_idx = 0
    shutdown_requested = False
    is_playing = False

    def print_current():
        file_path = audio_files[current_idx]
        console.print(
            f"\n[bold green]\\[CLIP {current_idx + 1}/{total_files}]:[/bold green] "
            f"[bold white]{file_path.name}[/bold white]"
        )

    console.print(
        Panel(
            f"[bold cyan]Instructions:[/bold cyan]\n"
            f"1. Press [bold green]SPACEBAR[/bold green] to play/stop the current clip.\n"
            f"2. Press [bold yellow]'n'[/bold yellow] or "
            f"[bold yellow]RIGHT[/bold yellow] for next clip.\n"
            f"3. Press [bold yellow]'p'[/bold yellow] or "
            f"[bold yellow]LEFT[/bold yellow] for previous clip.\n"
            f"4. Press [bold red]'q'[/bold red] to quit.\n\n"
            f"[bold cyan]Directory:[/bold cyan]\n"
            f"• [green]{dir_path}[/green] ({total_files} files)\n",
            title="🎧  [bold magenta]INTERACTIVE AUDIO PLAYER[/bold magenta]",
            border_style="cyan",
            expand=False,
        )
    )

    print_current()

    def play_audio():
        nonlocal is_playing
        sd.stop()
        try:
            data, fs = sf.read(audio_files[current_idx])
            sd.play(data, fs)
            is_playing = True
            console.print("[bold green]▶️  PLAYING[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error playing file: {e}[/bold red]")
            is_playing = False

    def stop_audio():
        nonlocal is_playing
        sd.stop()
        is_playing = False
        console.print("[bold bright_black]⏹️  STOPPED[/bold bright_black]")

    def on_press(key):
        nonlocal current_idx, shutdown_requested, is_playing

        if hasattr(key, "char") and key.char:
            char = key.char.lower()
            if char == "q":
                shutdown_requested = True
                return False
            elif char == "n":
                stop_audio()
                current_idx = (current_idx + 1) % total_files
                print_current()
                play_audio()
            elif char == "p":
                stop_audio()
                current_idx = (current_idx - 1) % total_files
                print_current()
                play_audio()

        if key == keyboard.Key.space:
            if is_playing:
                stop_audio()
            else:
                play_audio()
        elif key == keyboard.Key.right:
            stop_audio()
            current_idx = (current_idx + 1) % total_files
            print_current()
            play_audio()
        elif key == keyboard.Key.left:
            stop_audio()
            current_idx = (current_idx - 1) % total_files
            print_current()
            play_audio()

    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    new_attr = termios.tcgetattr(fd)
    new_attr[3] = new_attr[3] & ~termios.ECHO  # Disable ECHO

    try:
        termios.tcsetattr(fd, termios.TCSANOW, new_attr)
        with keyboard.Listener(on_press=on_press) as listener:
            while not shutdown_requested:
                time.sleep(0.1)
                # Check if audio finished playing naturally
                # In sounddevice, sd.wait() would block, so we just let it finish.
                pass
                if not listener.running:
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old_attr)
        sd.stop()

    console.print("\n[bold magenta]Session ended.[/bold magenta]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Audio Player")
    parser.add_argument(
        "--dir", default="data/synthetic_audio", help="Directory containing WAV files to review"
    )

    args = parser.parse_args()
    run_interactive_player(args.dir)
