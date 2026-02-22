import io
import wave
from pathlib import Path

import numpy as np

from db.models import Dataset, Record
from ui.core import db_client


def test_record_page_renders(test_client):
    response = test_client.get("/record")
    assert response.status_code == 200
    assert "Record Audio" in response.text
    assert "startRecording()" in response.text


def test_api_upload_audio_saves_file_and_tracks_db(test_client, tmp_path, monkeypatch):
    # Mock the out_dir so we don't write to real data/raw_audio during tests
    import ui.routes.record as rec_module

    monkeypatch.setattr(rec_module, "Path", lambda p: tmp_path if p.startswith("data") else Path(p))

    # Generate a dummy 1-second sine wave
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate, False)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)

    # Convert to WAV bytes in memory
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    wav_bytes = buf.getvalue()

    # Simulate multipart/form-data upload
    response = test_client.post(
        "/api/upload_audio",
        files={"audio": ("test.wav", wav_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    assert "Successfully saved and tracked" in response.text

    # Verify DB tracking
    with db_client.session_scope() as session:
        records = session.query(Record).filter_by(content="browser_recording").all()
        assert len(records) == 1
        assert records[0].data_type == "AUDIO"
        assert records[0].file_hash is not None

        ds = session.query(Dataset).filter_by(name="ui_voice_train").first()
        assert ds is not None
        assert len(ds.records) == 1
