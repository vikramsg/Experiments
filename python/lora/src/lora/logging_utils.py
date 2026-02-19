from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    handlers: list[logging.Handler] = [RichHandler(rich_tracebacks=True, show_path=False)]

    try:
        log_dir = Path("outputs")
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"run_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)
    except Exception:
        # Fallback to console-only if file logging fails (e.g., permissions)
        pass

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )

    # Log the command used to start the application
    logging.info("COMMAND: %s", " ".join(sys.argv))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
