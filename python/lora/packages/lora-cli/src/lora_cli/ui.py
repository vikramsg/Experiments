from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
import time

class VoiceUI:
    def __init__(self, console: Console):
        self.console = console
        self.status_panel = Panel("Idle", title="Status", border_style="blue")

    def print_system(self, message: str):
        self.console.print(f"[bold blue]System:[/bold blue] {message}")

    def print_user_message(self, message: str):
        self.console.print(f"\n[bold green]You:[/bold green] {message}\n")

    def print_error(self, message: str):
        self.console.print(f"[bold red]Error:[/bold red] {message}")

    def update_status(self, status: str, recording: bool = False):
        style = "bold red" if recording else "bold blue"
        self.console.print(f"[{style}]{status}[/{style}]", end="\r")

    def show_spinner(self, message: str):
        with self.console.status(f"[bold yellow]{message}[/bold yellow]", spinner="dots"):
            time.sleep(0.5) # UX pause

    def reset_status(self):
        self.console.print("[dim]Hold SPACE to speak[/dim]", end="\r")
