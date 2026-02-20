from __future__ import annotations

import json
from pathlib import Path

import pytest

from lora_data.data_loader import load_manifest, normalize_audio


def test_normalize_audio_accepts_array() -> None:
    assert normalize_audio([0.0, 1.0]) == [0.0, 1.0]


def test_normalize_audio_accepts_dict_array() -> None:
    assert normalize_audio({"array": [0.0]}) == [0.0]


def test_normalize_audio_rejects_bytes() -> None:
    with pytest.raises(ValueError):
        normalize_audio({"bytes": b"abc"})


def test_load_manifest_reads_lines(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"audio": [0.0], "text": "hi"}) + "\n")
    entries = load_manifest(manifest)
    assert entries[0]["text"] == "hi"
