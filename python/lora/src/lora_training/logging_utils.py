from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO, log_path: Path | None = None) -> None:
    root = logging.getLogger()
    # If handlers already exist, we clear them to reconfigure (e.g., to add a specific file handler)
    if root.handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    handlers: list[logging.Handler] = [RichHandler(rich_tracebacks=True, show_path=False)]

    try:
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            handlers.append(file_handler)
        else:
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
        pass

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )

    # Suppress noisy library logs
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Log the command used to start the application
    logging.info("COMMAND: %s", " ".join(sys.argv))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
