from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
