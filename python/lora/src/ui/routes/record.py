"""Interactive audio recording routes."""

import hashlib
import uuid
from pathlib import Path

from fasthtml.common import H2, Button, Div, P, Script, Style

from db.models import Dataset, DatasetRecord, Record
from ui.components.layout import PageLayout
from ui.core import db_client


def setup_routes(app, rt):
    @rt("/record")
    def get():
        # A simple UI to capture a single prompt.
        # Ideally, we would load the prompt list and show them one by one.
        # For now, a generic recorder for any prompt text.

        # We inject a small vanilla JS script to handle MediaRecorder
        js_code = """
        let mediaRecorder;
        let audioChunks = [];

        async function startRecording() {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                
                const formData = new FormData();
                formData.append('audio', audioBlob, 'recording.wav');
                
                const response = await fetch('/api/upload_audio', {
                    method: 'POST',
                    body: formData
                });
                
                const resultText = await response.text();
                document.getElementById('record-result').innerHTML = resultText;
                audioChunks = [];
            };

            mediaRecorder.start();
            document.getElementById('btn-start').disabled = true;
            document.getElementById('btn-stop').disabled = false;
            document.getElementById('record-status').innerText = '🔴 Recording...';
        }

        function stopRecording() {
            mediaRecorder.stop();
            document.getElementById('btn-start').disabled = false;
            document.getElementById('btn-stop').disabled = true;
            document.getElementById('record-status').innerText = 'Processing...';
        }
        """

        return PageLayout(
            H2("Record Audio"),
            P("Record your voice directly from the browser."),
            Div(
                Button("Start Recording", id="btn-start", onclick="startRecording()"),
                Button("Stop Recording", id="btn-stop", onclick="stopRecording()", disabled=True),
                Style("button { margin-right: 1rem; }"),
            ),
            P(id="record-status", style="margin-top: 1rem; font-weight: bold;"),
            Div(id="record-result", style="margin-top: 1rem;"),
            Script(js_code),
            title="Record - LoRA Studio",
        )

    @rt("/api/upload_audio")
    async def post(request):
        # We need to manually parse the form data because FastHTML's auto-injection
        # for binary uploads can be tricky depending on the exact Starlette version.
        form = await request.form()
        audio_file = form.get("audio")

        if not audio_file:
            return "No audio received.", 400

        audio_bytes = await audio_file.read()

        unique_id = uuid.uuid4().hex[:8]
        out_dir = Path("data/raw_audio/ui_recordings")
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / f"clip_{unique_id}.wav"

        with open(file_path, "wb") as f:
            f.write(audio_bytes)

        h = hashlib.sha256()
        h.update(audio_bytes)
        audio_hash = h.hexdigest()

        with db_client.session_scope() as session:
            dataset_name = "ui_voice_train"
            ds = session.query(Dataset).filter_by(name=dataset_name).first()
            if not ds:
                ds = Dataset(name=dataset_name, description="Browser recorded dataset")
                session.add(ds)
                session.flush()

            rec = Record(
                data_type="AUDIO",
                file_path=str(file_path),
                content="browser_recording",  # Later we can capture a prompt input
                file_hash=audio_hash,
                metadata_json={"source_type": "UI_REAL"},
            )
            session.add(rec)
            session.flush()

            session.add(DatasetRecord(dataset_id=ds.id, record_id=rec.id))

        return Div(
            P(f"✅ Successfully saved and tracked {len(audio_bytes)} bytes."),
            P(f"Saved to: {file_path}"),
        )
