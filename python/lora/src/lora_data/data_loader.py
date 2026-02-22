"""Dataset and manifest loading/preprocessing helpers for LoRA ASR training."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from datasets import Audio, Dataset, load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader

from db.client import DBClient
from db.models import Dataset as DbDataset
from db.models import DatasetRecord, Record
from lora_training.logging_utils import get_logger
from lora_training.model_utils import normalize_audio_rms


@dataclass
class DatasetConfig:
    dataset: str
    split: str
    max_samples: int | None
    max_seconds: float
    seed: int


def generate_tone(duration: float, sample_rate: int, freq: float) -> list[float]:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (0.1 * np.sin(2 * math.pi * freq * t)).astype(np.float32).tolist()


def build_synthetic_dataset(sample_rate: int, max_seconds: float) -> Dataset:
    phrases = [
        ("beep one", 440.0),
        ("beep two", 554.0),
        ("beep three", 659.0),
        ("beep four", 880.0),
        ("beep five", 988.0),
        ("beep six", 523.0),
        ("beep seven", 740.0),
        ("beep eight", 830.0),
    ]
    records: dict[str, list[Any]] = {"audio": [], "text": []}
    for text, freq in phrases:
        duration = max(1.0, min(max_seconds, 2.0))
        audio = generate_tone(duration, sample_rate, freq)
        records["audio"].append(audio)
        records["text"].append(text)
    return Dataset.from_dict(records)


def resample_audio(audio: list[float], original_rate: int, target_rate: int) -> list[float]:
    if original_rate != target_rate:
        raise ValueError(
            f"Mismatched sample rates: {original_rate} != {target_rate}. "
            "Resampling must be done explicitly by caller."
        )
    return audio


def load_librispeech_stream(split: str, sample_rate: int, max_samples: int | None) -> Dataset:
    LOGGER.info("Streaming LibriSpeech | split=%s | max_samples=%s", split, max_samples)
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split=split,
        streaming=True,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    if max_samples:
        dataset = dataset.take(max_samples)
    records: dict[str, list[Any]] = {"audio": [], "text": [], "speaker_id": []}
    for sample in dataset:
        audio_info = sample["audio"]
        audio_array = audio_info["array"]
        if isinstance(audio_array, np.ndarray) and audio_array.dtype == np.float64:
            audio_array = audio_array.astype(np.float32)

        # Audio feature handles resampling if sampling_rate is specified in cast_column.
        # Audio(sampling_rate=...) does resampling automatically on access.
        records["audio"].append(audio_array)
        records["text"].append(sample["text"])
        records["speaker_id"].append(sample.get("speaker_id", -1))
    return Dataset.from_dict(records)


def prepare_features(batch: dict[str, Any], processor: Any, sample_rate: int) -> dict[str, Any]:
    audio = normalize_audio_rms(batch["audio"], target_rms=0.075)
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    labels = processor.tokenizer(batch["text"], return_tensors="pt").input_ids

    if hasattr(inputs, "input_features"):
        input_key = "input_features"
        input_tensor = inputs.input_features[0]
    elif hasattr(inputs, "input_values"):
        input_key = "input_values"
        input_tensor = inputs.input_values[0]
    else:
        raise ValueError("Processor returned neither 'input_features' nor 'input_values'.")

    if not hasattr(inputs, "attention_mask") or inputs.attention_mask is None:
        raise ValueError("Processor did not return 'attention_mask'.")

    attention_mask = inputs.attention_mask[0]

    return {
        input_key: input_tensor,
        "attention_mask": attention_mask,
        "labels": labels[0],
    }


def to_tensor(values: Any) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values
    return torch.tensor(values)


def collate_features(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if "input_features" in features[0]:
        input_key = "input_features"
    elif "input_values" in features[0]:
        input_key = "input_values"
    else:
        raise KeyError("Features must contain 'input_features' or 'input_values'")

    input_values_list = [to_tensor(item[input_key]).squeeze() for item in features]
    attention_list = [to_tensor(item["attention_mask"]).squeeze() for item in features]
    labels_list = [to_tensor(item["labels"]) for item in features]
    input_lengths = [item.shape[-1] for item in input_values_list]
    max_input_len = max(input_lengths)
    input_batch = []
    attention_batch = []
    for input_values, attention_mask in zip(input_values_list, attention_list, strict=True):
        pad_len = max_input_len - input_values.shape[-1]
        if pad_len:
            input_values = F.pad(input_values, (0, pad_len))
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
        input_batch.append(input_values)
        attention_batch.append(attention_mask)

    label_lengths = [item.shape[-1] for item in labels_list]
    max_label_len = max(label_lengths)
    label_batch = []
    for labels in labels_list:
        pad_len = max_label_len - labels.shape[-1]
        if pad_len:
            labels = F.pad(labels, (0, pad_len), value=-100)
        label_batch.append(labels)

    return {
        input_key: torch.stack(input_batch),
        "attention_mask": torch.stack(attention_batch),
        "labels": torch.stack(label_batch),
    }


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_features,
    )


def split_by_speaker(
    dataset: Dataset, test_ratio: float, val_ratio: float, seed: int
) -> tuple[Dataset, Dataset, Dataset]:
    speaker_ids = sorted({item for item in dataset["speaker_id"]})
    if len(speaker_ids) < 3:
        raise ValueError(
            f"Not enough speakers to split by speaker id. Found {len(speaker_ids)}, required >= 3."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(speaker_ids)
    total = len(speaker_ids)
    test_count = max(1, int(total * test_ratio))
    val_count = max(1, int(total * val_ratio))
    test_ids = set(speaker_ids[:test_count])
    val_ids = set(speaker_ids[test_count : test_count + val_count])
    train_ids = set(speaker_ids[test_count + val_count :])

    def filter_by(ids: set[int]) -> Dataset:
        indices = [i for i, speaker in enumerate(dataset["speaker_id"]) if speaker in ids]
        return dataset.select(indices)

    train = filter_by(train_ids)
    val = filter_by(val_ids)
    test = filter_by(test_ids)
    LOGGER.info(
        "Speaker split | train=%s | val=%s | test=%s",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


def load_dataset_split(config: DatasetConfig, sample_rate: int) -> Dataset:
    match config.dataset:
        case "synthetic":
            LOGGER.info("Loading synthetic dataset | max_seconds=%s", config.max_seconds)
            return build_synthetic_dataset(sample_rate, config.max_seconds)
        case "librispeech_dummy":
            LOGGER.info("Loading dummy dataset | split=validation")
            dummy = load_dataset(
                "hf-internal-testing/librispeech_asr_dummy",
                "clean",
                split="validation",
            )
            records = {
                "audio": [item["audio"]["array"] for item in dummy],
                "text": [item["text"] for item in dummy],
                "speaker_id": [item.get("speaker_id", -1) for item in dummy],
            }
            return Dataset.from_dict(records)
        case "librispeech_clean":
            LOGGER.info(
                "Loading librispeech clean | split=%s | max_samples=%s",
                config.split,
                config.max_samples,
            )
            return load_librispeech_stream(config.split, sample_rate, config.max_samples)
        case _:
            raise ValueError(f"Unsupported dataset: {config.dataset}")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def normalize_audio(value: Any) -> list[float]:
    match value:
        case list():
            return value
        case dict() if "array" in value:
            return value["array"]
        case str():
            speech, _ = librosa.load(value, sr=16000)
            return speech.tolist()
        case _:
            raise ValueError(
                "Unsupported audio format in manifest. Expected array, list, or file path string."
            )


def build_manifest_dataset(entries: list[dict[str, Any]]) -> Dataset:
    """Create a dataset from manifest entries.

    Args:
        entries: Parsed manifest entries.

    Returns:
        Dataset containing audio, text, and speaker_id fields.
    """
    if not entries:
        raise ValueError("Manifest is empty")
    records = {
        "audio": [normalize_audio(item["audio"]) for item in entries],
        "text": [item["text"] for item in entries],
        "speaker_id": [item.get("speaker_id", -1) for item in entries],
    }
    return Dataset.from_dict(records)


def _load_db_dataset(dataset_name: str) -> Dataset:
    client = DBClient()

    with client.session_scope() as session:
        records = (
            session.query(Record)
            .join(DatasetRecord)
            .join(DbDataset)
            .filter(DbDataset.name == dataset_name, Record.is_valid)
            .all()
        )

        if not records:
            raise ValueError(f"No samples found for DB dataset: {dataset_name}")

        entries = [{"audio": r.file_path, "text": r.content, "speaker_id": -1} for r in records]

    return build_manifest_dataset(entries)


def _load_jsonl_dataset(path: Path) -> Dataset:
    return build_manifest_dataset(load_manifest(path))


def load_manifest_dataset(path: str | Path) -> Dataset:
    """Load a JSONL manifest or DB dataset into a Hugging Face dataset.

    Args:
        path: Path to the manifest file or db://<dataset_name>.

    Returns:
        Dataset containing audio, text, and speaker_id fields.
    """
    match str(path).split("://", 1):
        case ["db", dataset_name]:
            return _load_db_dataset(dataset_name)
        case _:
            return _load_jsonl_dataset(Path(path))


def prepare_dataset(dataset: Dataset, processor: Any) -> Dataset:
    sample_rate = processor.feature_extractor.sampling_rate
    return dataset.map(
        lambda batch: prepare_features(batch, processor, sample_rate),
        remove_columns=[col for col in dataset.column_names if col != "speaker_id"],
    )


LOGGER = get_logger(__name__)
