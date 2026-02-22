from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from lora_data.data_loader import (
    load_manifest,
    load_manifest_dataset,
    normalize_audio,
    prepare_dataset,
    split_by_speaker,
)


def test_split_by_speaker_returns_splits() -> None:
    records = {
        "audio": [[0.0] * 10, [0.0] * 10, [0.0] * 10],
        "text": ["a", "b", "c"],
        "speaker_id": [1, 2, 3],
    }
    dataset = Dataset.from_dict(records)
    train, val, test = split_by_speaker(dataset, test_ratio=0.2, val_ratio=0.2, seed=42)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


def test_prepare_dataset_returns_features() -> None:
    class DummyTokenizer:
        def __call__(self, text, return_tensors):
            _ = return_tensors
            return type("Out", (), {"input_ids": np.array([[1, 2, 3]])})()

    class DummyProcessor:
        def __init__(self) -> None:
            self.feature_extractor = type("Extractor", (), {"sampling_rate": 16000})
            self.tokenizer = DummyTokenizer()

        def __call__(self, audio, sampling_rate, return_tensors):
            _ = sampling_rate
            _ = return_tensors
            return type(
                "Out",
                (),
                {
                    "input_values": np.array([audio]),
                    "attention_mask": np.array([[1] * len(audio)]),
                },
            )()

    processor = DummyProcessor()
    records = {"audio": [[0.1] * 10], "text": ["hello"], "speaker_id": [1]}
    dataset = Dataset.from_dict(records)
    processed = prepare_dataset(dataset, processor)
    sample = processed[0]
    assert "input_values" in sample
    assert "attention_mask" in sample
    assert "labels" in sample


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


def test_load_manifest_dataset_invalid_path():
    with pytest.raises(ValueError, match="Invalid manifest path"):
        load_manifest_dataset("invalid_path.txt")
    with pytest.raises(ValueError, match="Invalid manifest path"):
        load_manifest_dataset("s3://bucket/data.jsonl")


def test_load_manifest_dataset_local_jsonl(tmp_path: Path):
    manifest = tmp_path / "valid.jsonl"
    manifest.write_text(json.dumps({"audio": [0.0], "text": "test_audio"}) + "\n")

    ds = load_manifest_dataset(manifest)
    assert len(ds) == 1
    assert ds[0]["text"] == "test_audio"
