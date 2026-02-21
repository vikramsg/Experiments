# Moonshine Voice CLI

An interactive, terminal-based voice interface for interacting with Moonshine LoRA models.

## Features

- **Live Transcription:** Speak naturally and see the text appear in real-time.
- **Push-to-Talk:** Hold the Spacebar to record, release to transcribe.
- **Visual Feedback:** A dynamic volume meter and status indicators in the terminal.
- **Model Support:** Works with any Hugging Face model ID (default: `UsefulSensors/moonshine-tiny`).

## Installation

This package is part of the `lora` monorepo. To install dependencies:

```bash
cd packages/lora-cli
uv sync
```

## Usage

### Interactive Mode (Real Microphone)

1.  Run the CLI:
    ```bash
    uv run moonshine
    ```
2.  **Hold SPACE** to speak.
3.  **Release SPACE** to transcribe.
4.  Press `Ctrl+C` to exit.

### Non-Interactive Mode (Simulation)

Useful for testing the pipeline without hardware.

```bash
uv run moonshine --non-interactive
```

## Configuration

- **Model ID:** You can specify a different model ID (e.g., your fine-tuned adapter merged model):
    ```bash
    uv run moonshine --model-id "path/to/your/merged_model"
    ```
