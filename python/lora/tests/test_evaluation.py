from __future__ import annotations

from lora.evaluation import summarize_losses


def test_summarize_losses_handles_empty() -> None:
    assert summarize_losses([]) == 0.0


def test_summarize_losses_returns_mean() -> None:
    assert summarize_losses([1.0, 2.0, 3.0]) == 2.0
