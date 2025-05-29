import time
from typing import Any

from langgraph.pregel import Pregel
from prompt_toolkit import prompt
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner


def chat_cli(agent: Pregel, chat_config: dict[str, Any]) -> None:
    """Test CLI that echoes user input with beautiful formatting."""

    # Initialize Rich console
    console = Console()

    # Initialize Prompt Toolkit with history
    history = InMemoryHistory()

    # Display welcome message
    console.print(
        Panel(
            "🧪 Chat CLI\nType 'exit', 'quit', or 'bye' to end",
            title="Welcome",
            border_style="green",
        )
    )

    while True:
        try:
            # Get user input with history and auto-suggest
            user_input = prompt(
                "💬 You: ", history=history, auto_suggest=AutoSuggestFromHistory(), multiline=False
            ).strip()

            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "bye", "q"]:
                console.print(Panel("👋 Goodbye! Thanks for chatting!", title="Farewell", border_style="yellow"))
                break

            if not user_input:
                continue

            console.print(Panel(f"You said: {user_input}", title="🤖 You", border_style="blue"))
            with Live(
                Spinner("dots", text="Processing your message..."),
                console=console,
                refresh_per_second=10,
                transient=True,
            ):
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=chat_config,  # type: ignore
                    stream_mode=["messages", "updates", "custom"],  # type: ignore
                )
                # TODO: Parse out thinking and response into separate messages
                # Have the parser as a function input to this function

            console.print(Panel(f"🤖 Bot: {result}", title="🤖 Bot", border_style="green"))

        except KeyboardInterrupt:
            console.print(Panel("\n👋 Chat interrupted. Goodbye!", title="Interrupted", border_style="red"))
            break
        except Exception as e:
            console.print(Panel(f"❌ Error occurred: {e}", title="Error", border_style="red"))
