from pathlib import Path

import numpy as np

from db.client import DBClient
from db.models import Dataset, Record
from lora_data.correction_recorder import save_correction


def test_save_correction_creates_db_entries(tmp_path: Path, monkeypatch):
    import lora_data.recorder as recorder_module

    db_file = tmp_path / "test.sqlite"
    client = DBClient(db_path=db_file)
    client.init_db()

    # Monkeypatch the DBClient used internally
    monkeypatch.setattr(recorder_module, "DBClient", lambda: client)

    out_dir = tmp_path / "audio"
    dataset_name = "test_corrections"
    sample_rate = 16000

    # Generate 0.5s of fake audio (440Hz sine wave)
    t = np.linspace(0, 0.5, int(sample_rate * 0.5), False)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16).reshape(-1, 1)

    corrected_text = "this is fake text"

    # Run the function
    saved_wav_path = save_correction(audio_data, corrected_text, out_dir, dataset_name, sample_rate)

    # 1. Assert WAV file was created
    assert saved_wav_path.exists()
    assert saved_wav_path.suffix == ".wav"
    assert saved_wav_path.parent == out_dir

    # 2. Verify DB entries
    with client.session_scope() as session:
        ds = session.query(Dataset).filter_by(name=dataset_name).first()
        assert ds is not None

        records = session.query(Record).all()
        assert len(records) == 1
        assert records[0].content == corrected_text
        assert records[0].file_path == str(saved_wav_path)
        assert records[0].data_type == "AUDIO"


def test_save_correction_appends_multiple(tmp_path: Path, monkeypatch):
    import lora_data.recorder as recorder_module

    db_file = tmp_path / "test.sqlite"
    client = DBClient(db_path=db_file)
    client.init_db()
    monkeypatch.setattr(recorder_module, "DBClient", lambda: client)

    out_dir = tmp_path / "audio"
    dataset_name = "test_corrections"
    audio_data = np.zeros((1600, 1), dtype=np.int16)

    save_correction(audio_data, "first", out_dir, dataset_name)
    save_correction(audio_data, "second", out_dir, dataset_name)

    with client.session_scope() as session:
        records = session.query(Record).all()
        assert len(records) == 2
        assert records[0].content == "first"
        assert records[1].content == "second"
