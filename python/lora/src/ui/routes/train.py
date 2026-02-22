"""Training orchestration routes."""

import subprocess

from fasthtml.common import H2, H3, Button, Div, Form, Input, Label, P

from db.models import Run
from ui.components.experiment import LogViewer, StatusBadge
from ui.components.layout import PageLayout
from ui.core import db_client


def setup_routes(app, rt):
    @rt("/train")
    def get():
        return PageLayout(
            H2("Start Training Experiment"),
            P("Configure and launch a LoRA fine-tuning run."),
            Form(
                Label(
                    "Experiment Name",
                    Input(type="text", name="run_name", value="my_voice_tune_v5", required=True),
                ),
                Label(
                    "Max Steps", Input(type="number", name="max_steps", value="1000", required=True)
                ),
                Label(
                    "Learning Rate",
                    Input(type="text", name="learning_rate", value="1e-4", required=True),
                ),
                Label(
                    "Adapter Config",
                    Input(
                        type="text",
                        name="adapter_args",
                        value="--lora-r 32 --lora-alpha 64 --lora-dropout 0.1 --use-dora",
                    ),
                ),
                Button("Start Training", type="submit"),
                hx_post="/api/train",
                hx_target="#train-status",
                hx_indicator="#train-loading",
            ),
            Div(id="train-loading", cls="htmx-indicator", style="display:none;"),
            Div(id="train-status", style="margin-top: 2rem;"),
            title="Train - LoRA Studio",
        )

    @rt("/api/train")
    def post(run_name: str, max_steps: str, learning_rate: str, adapter_args: str):
        import os

        out_dir = f"outputs/{run_name}"
        os.makedirs(out_dir, exist_ok=True)
        log_file = f"{out_dir}/experiment.log"

        with db_client.session_scope() as session:
            # We assume experiment already exists or we just create a run
            run = Run(
                name=run_name,
                run_type="TRAINING",
                status="RUNNING",
                log_file_path=log_file,
                artifacts_dir=out_dir,
            )
            session.add(run)
            session.flush()
            run_id = run.id

        cmd = [
            "uv",
            "run",
            "python",
            "main.py",
            "--output-dir",
            out_dir,
            "--max-steps",
            max_steps,
            "--learning-rate",
            learning_rate,
        ]
        if adapter_args:
            cmd.extend(adapter_args.split())

        with open(f"{out_dir}/nohup.out", "w") as out:
            # Start background subprocess
            p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)

        return Div(
            H3(f"Experiment: {run_name}", StatusBadge("RUNNING")),
            P(f"Process PID: {p.pid}"),
            # This triggers HTMX to poll the logs endpoint every 2 seconds
            Div(hx_get=f"/api/train/status/{run_id}", hx_trigger="every 2s"),
        )

    @rt("/api/train/status/{run_id}")
    def get_status(run_id: int):
        with db_client.session_scope() as session:
            run = session.query(Run).filter_by(id=run_id).first()
            if not run:
                return "Run not found"
            return LogViewer(run.log_file_path)
