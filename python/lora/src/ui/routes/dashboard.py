"""Dashboard routes."""

from fasthtml.common import H2, H3, A, Code, Div, Footer, P, Small, Table, Tbody, Td, Th, Thead, Tr
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from db.models import Experiment
from ui.components.layout import PageLayout
from ui.core import db_client


def setup_routes(app, rt):
    @rt("/")
    def get():
        rows = []
        with db_client.session_scope() as session:
            # Load experiments and their runs for summary
            stmt = (
                select(Experiment)
                .options(selectinload(Experiment.runs))
                .order_by(Experiment.created_at.desc())
            )
            experiments = session.execute(stmt).scalars().all()

            for exp in experiments:
                runs_count = len(exp.runs)
                rows.append(
                    Tr(
                        Td(exp.name),
                        Td(exp.description or "-"),
                        Td(exp.created_at.strftime("%Y-%m-%d %H:%M:%S")),
                        Td(str(runs_count)),
                        Td(A("View", href=f"/experiments/{exp.id}")),
                    )
                )

        if not rows:
            exp_content = P("No experiments found. Start training to create an experiment.")
        else:
            exp_content = Table(
                Thead(Tr(Th("Name"), Th("Description"), Th("Created"), Th("Runs"), Th("Action"))),
                Tbody(*rows),
            )

        db_path_str = str(db_client.db_path.absolute())

        # Styles broken down to fit line length limits
        footer_style = (
            "margin-top: 4rem; "
            "padding-top: 1rem; "
            "border-top: 1px solid var(--pico-muted-border-color);"
        )

        return PageLayout(
            Div(
                H2("Dashboard"),
                P("Welcome to the LoRA training interface. Select a tool from the menu."),
                H3("Experiments", style="margin-top: 2rem;"),
                exp_content,
                Footer(
                    Small("Database path: ", Code(db_path_str)),
                    style=footer_style,
                ),
            )
        )
