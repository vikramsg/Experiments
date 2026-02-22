"""Synthetic data generation routes."""

import multiprocessing

from fasthtml.common import H2, H3, Button, Div, Form, Input, Label, P

from lora_data.generate_synthetic_data import generate_synthetic_data
from ui.components.layout import PageLayout


def _run_generation(num_samples: int, dataset_name: str, audio_prefix: str, mix_with_real: bool):
    """Run generation in a background process to avoid blocking the UI."""
    try:
        generate_synthetic_data(
            num_samples=num_samples,
            dataset_name=dataset_name,
            audio_prefix=audio_prefix,
            seed=42,
        )
        # TODO: Handle mixing with real data if mix_with_real is True
        # For now, just logging completion is enough
    except Exception:
        import traceback

        traceback.print_exc()


def setup_routes(app, rt):
    @rt("/generate")
    def get():
        return PageLayout(
            H2("Synthetic Data Generation"),
            P("Generate synthetic voice datasets using F5-TTS."),
            Form(
                Label(
                    "Number of Samples",
                    Input(type="number", name="num_samples", value="5", required=True),
                ),
                Label(
                    "Dataset Name (DB)",
                    Input(type="text", name="dataset_name", value="synthetic_train", required=True),
                ),
                Label(
                    "Audio Prefix",
                    Input(type="text", name="audio_prefix", value="synth_gen", required=True),
                ),
                Label(
                    Input(type="checkbox", name="mix_with_real", value="true"),
                    " Mix with real_voice_train",
                ),
                Button("Generate", type="submit"),
                hx_post="/api/generate",
                hx_target="#generate-result",
                hx_indicator="#generate-loading",
            ),
            Div(id="generate-loading", cls="htmx-indicator", style="display:none;"),
            Div(id="generate-result", style="margin-top: 2rem;"),
            title="Generate Data - LoRA Studio",
        )

    @rt("/api/generate")
    def post(num_samples: int, dataset_name: str, audio_prefix: str, mix_with_real: str = "false"):
        mix_bool = mix_with_real.lower() == "true"

        # Start the generation in a separate process so we can return the UI immediately
        p = multiprocessing.Process(
            target=_run_generation, args=(num_samples, dataset_name, audio_prefix, mix_bool)
        )
        p.start()

        return Div(
            H3("Generation Started!"),
            P(
                f"Generating {num_samples} samples into dataset '{dataset_name}'. "
                "Check the terminal for progress."
            ),
            P(f"Process ID: {p.pid}"),
            cls="success",
        )
