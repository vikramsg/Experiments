"""Reusable FastHTML layouts and containers."""

from fasthtml.common import H1, A, Container, Div, Header, Li, Nav, Title, Ul


def PageLayout(*children, title="LoRA Studio"):
    """Base layout wrapping every page."""
    return (
        Title(title),
        Container(
            Header(
                Nav(
                    Ul(
                        Li(
                            A(
                                H1("LoRA Studio", style="margin-bottom: 0; font-size: 1.5rem;"),
                                href="/",
                                cls="contrast",
                            )
                        ),
                    ),
                    Ul(
                        Li(A("Datasets", href="/datasets")),
                        Li(A("Record", href="/record")),
                        Li(A("Generate", href="/generate")),
                        Li(A("Train", href="/train")),
                    ),
                ),
                style=(
                    "padding-bottom: 2rem; border-bottom: 1px solid var(--pico-muted-border-color);"
                    " margin-bottom: 2rem;"
                ),
            ),
            # The actual page content
            Div(*children),
        ),
    )
