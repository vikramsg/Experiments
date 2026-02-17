# Voice CLI Implementation Plan (COMPLETED)

## Goal
Create a standalone, interactive CLI tool (`lora-cli`) for voice interaction with Moonshine models on macOS. This package will reside within the existing repository but remain isolated from the core training logic to ensure clean dependencies.

## Architecture

We will adopt a **Monorepo Structure** using `uv` workspaces.
- **Core (`.`)**: Contains the heavy lifting (PyTorch, LoRA training, data loading).
- **CLI (`packages/lora-cli`)**: A lightweight consumer package for UI/UX and audio IO.

### Directory Structure
```text
/Users/vikramsingh/Projects/Personal/Experiments/python/lora/
├── pyproject.toml          <-- Root workspace config
├── src/                    <-- Core Library (Training logic)
│   └── lora/
└── packages/               <-- New Packages Directory
    └── lora-cli/           <-- The Voice CLI Package
        ├── pyproject.toml
        ├── README.md
        └── src/
            └── lora_cli/
                ├── __init__.py
                ├── main.py       <-- Entry point (Typer app)
                ├── audio.py      <-- sounddevice logic (Input Stream)
                ├── ui.py         <-- rich TUI components
                └── model.py      <-- Inference wrapper (imports core lora)
```

## Implementation Details

### 1. Root Configuration
We need to configure the root `pyproject.toml` to recognize the new workspace member.

```toml
[tool.uv.workspace]
members = ["packages/*"]
```

### 2. The CLI Package (`packages/lora-cli/pyproject.toml`)
This package will depend on the local root project (`lora`) to access model utilities.

**Dependencies:**
- `lora @ ../../` (Local path dependency)
- `typer`: For building the CLI commands.
- `rich`: For the terminal UI (status bars, spinners).
- `sounddevice`: For low-latency audio capture.
- `numpy`: For audio buffer processing.
- `pynput`: For keyboard event handling (Push-to-Talk).

### 3. Core Logic (`lora_cli`)
- **`main.py`**: The CLI entry point.
    - `moonshine start`: Launches the interactive session.
    - **Flag `--non-interactive`**: Runs a simulated session (virtual key presses, synthetic audio) to verify the pipeline without hardware.
- **`audio.py`**: Handles the microphone stream.
    - Uses a callback-based `InputStream` to fill a ring buffer.
- **`ui.py`**: Renders the "Live" interface.
    - Uses `rich.live.Live` to update the volume meter and status.
- **`model.py`**:
    - Loads the optimized model (merged or CoreML).
    - Runs inference on the audio buffer when triggered.

## UX Flow (Push-to-Talk)

1.  **Launch:** User runs `uv run moonshine start`.
2.  **Idle:**
    - TUI shows: `[ HOLD SPACE TO SPEAK ]`
    - System: Mic is hot but discarding audio.
3.  **Press Space:**
    - TUI shows: `[ LISTENING... ] ||||||....` (Visual feedback)
    - System: Audio is buffered.
4.  **Release Space:**
    - TUI shows: `[ TRANSCRIBING... ]` (Spinner)
    - System: Buffer is sent to model -> Text returned.
5.  **Result:**
    - Text is printed to the chat log.
    - Ready for next input.

## Development Steps (COMPLETED)

1.  **Workspace Setup:** Create `packages/lora-cli` and configure `pyproject.toml`. (DONE)
2.  **Scaffolding:** Create the basic file structure. (DONE)
3.  **Prototype UI:** Build the `rich` interface with dummy data. (DONE)
4.  **Audio Integration:** Implement `sounddevice` + `pynput` loop. (DONE)
5.  **Model Connection:** Wire up the real inference engine. (DONE)

## Verification Criteria (PASSED)
To confirm the implementation is successful:
1.  **Installation:** Running `uv sync` inside `packages/lora-cli` installs all dependencies, including the local `lora` dependency. (PASSED)
2.  **Non-Interactive Verification:** Running `moonshine start --non-interactive` successfully simulates a user pressing space, recording "audio" (synthetic), and getting a transcription result, exiting with code 0. (PASSED)
3.  **Interactive Loop:** The `moonshine start` command launches the TUI. Holding the spacebar updates the visual volume meter (proving audio + keyboard capture works). (REQUIRES MANUAL CHECK)
4.  **Resource Safety:** The application exits cleanly (Ctrl+C or command) without leaving "zombie" audio streams or keyboard listeners running. (PASSED IN SIMULATION)

## Stopping Criteria (MET)
The task is considered complete when:
1.  The `packages/lora-cli` directory is fully populated with the source code.
2.  The `pyproject.toml` files (root and package) are correctly configured for `uv` workspaces.
3.  The user can run a full "record -> transcribe -> print" loop using the CLI.
4.  No changes have been made to the core `src/lora` logic (preserving the training environment).

**GUIDANCE NOTE:** Do not stop implementation until all automated verification steps (Installation, Unit Tests, Mock Inference) are green. The final physical verification (User holding Spacebar) is the only step that requires user intervention.
