# FastHTML UI Transition Plan

**Goal**: Replace all CLI-based interactions with a unified, lightweight, pure-Python web dashboard using FastHTML, backed by the existing SQLite `tracker.db`.

**Constraints**:
*   No heavy frontend frameworks (React, Vue, Streamlit).
*   No Node.js or Javascript build steps.
*   Must be highly modular (no monolithic `app.py`).
*   Must be fully end-to-end testable without manual browser interaction.
*   Must maintain strict linting (Ruff) and testing (Pytest) standards.

### Phase 1: Infrastructure & Scaffolding (The Foundation)
1.  **Dependencies**: Add `python-fasthtml` and `pytest-asyncio` to a new `[dependency-groups.ui]` section in `pyproject.toml`.
2.  **Tasks**: Update `justfile` to include `ui-sync` (syncs the `ui` group), `ui-dev` (runs the FastHTML server with live reload), and `ui-test` (runs `pytest tests/ui/`).
3.  **Directory Structure**:
    *   Create `src/ui/__init__.py`, `src/ui/main.py` (FastHTML app entrypoint).
    *   Create `src/ui/core.py` (Database client initialization for the web context).
    *   Create `src/ui/components/__init__.py`, `src/ui/components/layout.py` (Base HTML shell, PicoCSS CDN links, Navbar).
    *   Create `src/ui/routes/__init__.py`, `src/ui/routes/dashboard.py` (The `/` route).
4.  **Testing Scaffold**:
    *   Create `tests/ui/__init__.py`, `tests/ui/conftest.py`.
    *   In `conftest.py`, define a `test_client` fixture that returns a FastHTML `Client(app)` connected to an in-memory SQLite database (`sqlite:///:memory:`).
    *   Create `tests/ui/test_dashboard.py` asserting a 200 OK on `/`.
5.  **Verification (MANDATORY)**: Run `just lint`, `just fix`, and `just test` (including the new UI tests) to ensure the scaffold is solid.

### Phase 2: Read-Only Views (Data Explorer)
1.  **Components (`src/ui/components/dataset.py`)**: Build pure Python functions returning FastHTML `Table`, `Tr`, `Td` tags to render `Dataset`, `Run`, and `Record` objects from the database.
2.  **Routes (`src/ui/routes/datasets.py`)**: Implement `GET /datasets` to query the DB and return the dataset tables.
3.  **Audio Review**: Implement a route that serves static `.wav` files from the `data/` directory so they can be played in the browser using the native HTML `<audio controls>` tag.
4.  **Tests (`tests/ui/test_datasets.py`)**: Insert mock datasets into the test DB and assert the `Client(app).get("/datasets")` returns HTML containing those dataset names.
5.  **Verification (MANDATORY)**: Run `just lint`, `just fix`, and `just test`.

### Phase 3: Interactive Audio Generation & Recording (The Hard Part)
1.  **Synthetic Data Form (`src/ui/routes/generate.py`)**:
    *   Build an HTMX form (`Form(hx_post="/api/generate", hx_target="#result")`) capturing `num_samples`, `audio_prefix`, and a `mix-with-real` toggle.
    *   **The Route**: On POST, invoke the existing logic from `generate_synthetic_data.py`. FastHTML handles the background task, returning an HTMX progress/status indicator (`Div("Generation Started...", id="result")`).
2.  **Microphone Recorder (`src/ui/routes/record.py`)**:
    *   Build the UI with a "Record" button.
    *   Inject a tiny `<script>` block using the browser's `MediaRecorder` API to capture the microphone and `POST` the Blob to `/api/upload_audio`.
    *   **The Route**: Accept the audio bytes, hash them, save to disk (`data/raw_audio/`), and write the `Record` to the DB.
3.  **Headless Tests**:
    *   `test_generate.py`: Mock the `F5TTSEngine` so it doesn't spin up MLX, POST to `/api/generate`, and assert DB `Run` creation.
    *   `test_record.py`: Generate a 1-second 440Hz numpy sine wave (reusing `recorder.py` logic), convert to bytes, POST to `/api/upload_audio`, and assert DB `Record` creation.
4.  **Verification (MANDATORY)**: Run `just lint`, `just fix`, and `just test`.

### Phase 4: Training Orchestration
1.  **Experiment Form (`src/ui/routes/train.py`)**:
    *   Build an HTMX form capturing `ExperimentConfig` parameters (LR, steps, adapter config).
2.  **Background Process**:
    *   On POST `/api/train`, use `subprocess.Popen(["uv", "run", "python", "main.py", ...])` and store the PID/Run ID in the database.
    *   Return an HTMX polling element: `Div(hx_get=f"/api/train/status/{run_id}", hx_trigger="every 2s")`.
3.  **Live Logs (`src/ui/components/experiment.py`)**:
    *   The `/api/train/status` route reads the tail of `outputs/{run_name}/experiment.log` and returns it as a `<pre><code>` block.
4.  **Tests (`tests/ui/test_train.py`)**:
    *   Mock `subprocess.Popen` to write a dummy log file and exit immediately.
    *   POST to `/api/train`.
    *   GET the returned polling URL and assert the dummy logs are present in the HTML response.
5.  **Verification (MANDATORY)**: Run `just lint`, `just fix`, and `just test`.
