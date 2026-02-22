"""Dashboard routes."""

from fasthtml.common import H2, P

from ui.components.layout import PageLayout


def setup_routes(app, rt):
    @rt("/")
    def get():
        return PageLayout(
            H2("Dashboard"),
            P("Welcome to the LoRA training interface. Select a tool from the navigation menu."),
        )
