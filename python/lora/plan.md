# Refactoring and Tooling Plan for Personalized ASR Fine-Tuning

## Phase 1: Requirements Understanding

**Goal**: Refactor the codebase to split the monolith (`src/lora`) into clear domains (`src/lora_training`, `src/lora_data`) while completely ignoring the `lora-cli` package. Introduce a robust data collection workflow (`record_dataset.py`, `split_manifest.py`) and restructure dependencies into explicit groups within `pyproject.toml`.

**Execution Guidance (CRITICAL):**
The agent MUST NOT stop, ask for clarification, or request intermediate approval during execution. The agent is strictly mandated to execute this plan end-to-end autonomously. If errors occur, the agent must independently diagnose, fix, and resume the pipeline until all verification and acceptance criteria are completely fulfilled.

**Constraints**:
- Adhere strictly to the project's use of `uv` and `just`.
- Rely entirely on Python `logging` instead of `print` statements.
- Ensure all new public APIs and functions have proper type hints and Google-style docstrings.
- Python module names must use underscores (`_`), not hyphens (`-`).
- Dependencies must be strictly segregated by domain.

## Phase 2: Refactoring Core Packages & Dependencies

### Step 2.1: Reorganize Dependencies (`pyproject.toml`)
Extract specialized libraries from the core `dependencies` block into new dependency groups within `pyproject.toml`.

*   **`dependencies` (Core):** `torch`, `transformers`, `moonshine-voice`, `rich`, `soundfile`
*   **`[dependency-groups.training]`:** `peft`, `accelerate`, `evaluate`, `jiwer`, `datasets[audio]`
*   **`[dependency-groups.data]`:** `librosa`, `sounddevice`, `pynput`, `numpy`
*   **`[dependency-groups.dev]`:** (Existing: `ruff`, `pytest`)

### Step 2.2: Directory Splitting & Migration
Rename `src/lora` and create dedicated namespaces for `training` and `data`.

1.  Rename `src/lora` -> `src/lora_training`
2.  Create `src/lora_data` and copy necessary `__init__.py`.
3.  Move files:
    *   **To `src/lora_training`:** `evaluation.py`, `model_utils.py`, `training_config.py`, `runners/`
    *   **To `src/lora_data`:** `data_loader.py`, `manifest_diagnostics.py`
4.  Relocate Scripts: Move data-related standalone scripts from `scripts/` into the `src/lora_data/` module as functional code (e.g., `build_manifest.py` -> `src/lora_data/manifest_builder.py`).

### Step 2.3: Fix Imports & Entrypoints
Update all internal imports across the codebase to reflect the new `lora_training` and `lora_data` namespaces.

*   Fix `main.py` entrypoint.
*   Fix `tests/test_data_loader.py` and `tests/test_evaluation.py`.
*   Fix the test configurations if necessary.

## Phase 3: Data Collection Tooling & Interactive Workflow

Develop the tools described in `docs/personalized/tooling.md` within the `src/lora_data` namespace, focusing on a seamless interactive UX.

### Step 3.1: Prompt Sanitization
Create a quick script/routine to parse `docs/personalized/prompts.txt`, strip out conversational chat logs, and output a clean list of technical commands into `data/coding_prompts.txt`.

### Step 3.2: Build the Audio Recorder (`src/lora_data/recorder.py`)
Implement an interactive recording loop that uses `sounddevice` and `pynput` with the following workflow:
1.  **Prompt Selection:** Loads random phrases from `data/coding_prompts.txt`.
2.  **Display:** Prints the target phrase clearly to the terminal (e.g., `[PROMPT]: "revert that last commit"`).
3.  **Capture:** Waits for the user to press and hold `Spacebar`. Prints `🔴 RECORDING...`. Captures audio from the microphone while the key is held.
4.  **Save & Manifest:** On release, immediately saves the 16kHz, Mono, 16-bit PCM audio to `data/raw_audio/my_voice/clip_uuid.wav`. Appends the `{"audio": "...", "text": "..."}` entry directly to `data/my_voice_all.jsonl`.
5.  **Iteration/Correction:** Presents the next prompt automatically. Allows pressing `r` to discard the last recording and try again, or `q` to quit.

### Step 3.3: Headless Verification Mode
Add a `--non-interactive` flag to the recorder to enable automated testing without human intervention.
*   Bypasses the microphone capture and keyboard listener entirely.
*   Programmatically generates 1-2 seconds of synthetic 16kHz sine wave audio (using `numpy`).
*   Runs through the exact file saving and manifest appending logic to simulate a successful recording session.

### Step 3.4: Build the Manifest Splitter (`src/lora_data/manifest_utils.py`)
Create a utility function to split the master `my_voice_all.jsonl` into an 80% train / 20% eval split, ensuring random distribution.

### Step 3.5: Update Workflows (`justfile`)
Add new tasks to the `justfile`:
*   `just sync`: Update to handle `uv sync --all-groups`.
*   `just record-dataset`: Start the interactive voice recording session.
*   `just verify-recorder`: Run the headless verification mode.
*   `just split-dataset`: Split the master manifest.

## Phase 4: Validation & Acceptance

1.  **Dependency Isolation:** Verify that `uv run --group training python -c "import peft"` works, while `uv run --group data python -c "import peft"` fails (or similar validation).
2.  **Linting & Tests:** Ensure `just test` and `just lint` pass cleanly with no import errors.
3.  **Automated Recorder Verification (No Human Intervention):** Execute `just verify-recorder`. A test script must programmatically assert that:
    *   The headless recording session ran without blocking.
    *   A dummy `.wav` file was produced in the target directory.
    *   The `.wav` file properties exactly match requirements (16kHz, mono, 16-bit PCM).
    *   The dummy `manifest.jsonl` has a valid, properly formatted JSONL entry appended.
4.  **Training Compatibility:** Verify that `just run-experiment` still successfully boots up the `src/lora_training/runners/experiment.py` script.