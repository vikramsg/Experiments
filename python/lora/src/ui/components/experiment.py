"""Experiment rendering components."""

import os

from fasthtml.common import H3, Code, Div, P, Pre


def StatusBadge(status: str):
    """Render a colored badge based on status."""
    color = "grey"
    if status == "RUNNING":
        color = "blue"
    elif status == "COMPLETED":
        color = "green"
    elif status == "FAILED":
        color = "red"

    return span(
        status,
        style=f"background-color: {color}; color: white; padding: 2px 8px; "
        "border-radius: 4px; font-size: 0.8rem;",
    )


def LogViewer(log_path: str):
    """Render the tail of a log file, or a placeholder if missing."""
    if not log_path or not os.path.exists(log_path):
        return Div(
            P("Logs not available or experiment hasn't started yet.", cls="dim"), id="log-viewer"
        )

    try:
        # Just grab the last 50 lines for UI responsiveness
        with open(log_path) as f:
            lines = f.readlines()
            tail = "".join(lines[-50:])

        return Div(
            H3("Live Logs (Tail)"),
            Pre(Code(tail, style="font-size: 0.8rem; overflow-x: auto;")),
            id="log-viewer",
        )
    except Exception as e:
        return Div(P(f"Error reading logs: {e}"), id="log-viewer")


def span(*args, **kwargs):
    from fasthtml.common import ft

    return ft("span", *args, **kwargs)
