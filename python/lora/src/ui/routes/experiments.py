"""Experiment detail routes."""

from fasthtml.common import H2, H3, H4, A, Code, Div, P, Table, Tbody, Td, Th, Thead, Tr
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from db.models import Experiment, Run
from ui.components.layout import PageLayout
from ui.core import db_client


def setup_routes(app, rt):
    @rt("/experiments/{exp_id}")
    def get(exp_id: int):
        with db_client.session_scope() as session:
            stmt = (
                select(Experiment)
                .options(
                    selectinload(Experiment.runs).selectinload(Run.params),
                    selectinload(Experiment.runs).selectinload(Run.metrics),
                )
                .filter(Experiment.id == exp_id)
            )
            exp = session.execute(stmt).scalar_one_or_none()

            if not exp:
                return PageLayout(
                    Div(
                        H2("Experiment Not Found"),
                        P(f"No experiment exists with ID {exp_id}."),
                        A("Back to Dashboard", href="/", role="button"),
                    )
                )

            # Build Runs section
            runs_content = []
            if not exp.runs:
                runs_content.append(P("No runs recorded for this experiment."))
            else:
                for run in exp.runs:
                    # Params table
                    params_rows = [Tr(Td(Code(p.key)), Td(p.value)) for p in run.params]
                    if params_rows:
                        params_table = Table(
                            Thead(Tr(Th("Parameter"), Th("Value"))),
                            Tbody(*params_rows),
                            cls="striped",
                        )
                    else:
                        params_table = P("No parameters recorded.")

                    # Metrics table
                    metrics_rows = [
                        Tr(Td(Code(m.key)), Td(f"{m.value:.4f}"), Td(str(m.step)))
                        for m in run.metrics
                    ]
                    if metrics_rows:
                        metrics_table = Table(
                            Thead(Tr(Th("Metric"), Th("Value"), Th("Step"))),
                            Tbody(*metrics_rows),
                            cls="striped",
                        )
                    else:
                        metrics_table = P("No metrics recorded.")

                    run_style = (
                        "margin-bottom: 3rem; "
                        "padding: 1rem; "
                        "border: 1px solid var(--pico-muted-border-color); "
                        "border-radius: var(--pico-border-radius);"
                    )

                    run_div = Div(
                        H4(f"Run: {run.name}"),
                        P(f"Status: {run.status} | Type: {run.run_type}"),
                        P("Artifacts: ", Code(run.artifacts_dir or "None")),
                        Div(
                            Div(H4("Parameters"), params_table, cls="grid-item"),
                            Div(H4("Metrics"), metrics_table, cls="grid-item"),
                            cls="grid",  # Pico CSS grid
                        ),
                        style=run_style,
                    )
                    runs_content.append(run_div)

            return PageLayout(
                Div(
                    A("← Back to Dashboard", href="/"),
                    H2(f"Experiment: {exp.name}", style="margin-top: 1rem;"),
                    P(exp.description or ""),
                    P(f"Created at: {exp.created_at.strftime('%Y-%m-%d %H:%M:%S')}"),
                    H3("Runs", style="margin-top: 2rem;"),
                    *runs_content,
                )
            )
