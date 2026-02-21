import json
import wave
from pathlib import Path

import numpy as np

from lora_data.correction_recorder import save_correction


def test_save_correction_creates_files(tmp_path: Path):
    """Test that save_correction writes a valid WAV and appends to JSONL."""
    out_dir = tmp_path / "audio"
    manifest_path = tmp_path / "manifest.jsonl"

    # Generate 0.5s of fake audio (440Hz sine wave)
    sample_rate = 16000
    t = np.linspace(0, 0.5, int(sample_rate * 0.5), False)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16).reshape(-1, 1)

    corrected_text = "this is fake text"

    # Run the function
    saved_wav_path = save_correction(
        audio_data, corrected_text, out_dir, manifest_path, sample_rate
    )

    # 1. Assert WAV file was created
    assert saved_wav_path.exists()
    assert saved_wav_path.suffix == ".wav"
    assert saved_wav_path.parent == out_dir

    # 2. Verify WAV file is valid and has expected properties
    with wave.open(str(saved_wav_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # 16-bit
        assert wf.getframerate() == sample_rate
        assert wf.getnframes() == len(audio_data)

    # 3. Assert manifest was created and appended
    assert manifest_path.exists()

    with open(manifest_path) as f:
        lines = f.readlines()

    assert len(lines) == 1
    entry = json.loads(lines[0])

    # 4. Verify manifest contents
    assert entry["text"] == corrected_text
    assert Path(entry["audio"]).resolve() == saved_wav_path.resolve()


def test_save_correction_appends_multiple(tmp_path: Path):
    """Test that multiple calls append to the same manifest without overwriting."""
    out_dir = tmp_path / "audio"
    manifest_path = tmp_path / "manifest.jsonl"

    audio_data = np.zeros((1600, 1), dtype=np.int16)

    save_correction(audio_data, "first", out_dir, manifest_path)
    save_correction(audio_data, "second", out_dir, manifest_path)

    with open(manifest_path) as f:
        lines = f.readlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "first"
    assert json.loads(lines[1])["text"] == "second"
