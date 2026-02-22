"""Dataset and manifest loading/preprocessing helpers for LoRA ASR training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from datasets import Dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader

from db.client import DBClient
from db.models import Dataset as DbDataset
from db.models import DatasetRecord, Record
from lora_training.logging_utils import get_logger
from lora_training.model_utils import normalize_audio_rms


def resample_audio(audio: list[float], original_rate: int, target_rate: int) -> list[float]:
    if original_rate != target_rate:
        raise ValueError(
            f"Mismatched sample rates: {original_rate} != {target_rate}. "
            "Resampling must be done explicitly by caller."
        )
    return audio


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
    """Create a dataset from manifest entries."""
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
    """Load a JSONL manifest or DB dataset into a Hugging Face dataset."""
    match str(path).split("://", 1):
        case ["db", dataset_name]:
            return _load_db_dataset(dataset_name)
        case [local_path] if local_path.endswith(".jsonl"):
            return _load_jsonl_dataset(Path(local_path))
        case _:
            raise ValueError(
                f"Invalid manifest path: '{path}'. Expected 'db://<name>' or a '.jsonl' file."
            )


def prepare_dataset(dataset: Dataset, processor: Any) -> Dataset:
    sample_rate = processor.feature_extractor.sampling_rate
    return dataset.map(
        lambda batch: prepare_features(batch, processor, sample_rate),
        remove_columns=[col for col in dataset.column_names if col != "speaker_id"],
    )


LOGGER = get_logger(__name__)
