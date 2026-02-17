from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader


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
    if original_rate == target_rate:
        return audio
    resampled = librosa.resample(np.asarray(audio), orig_sr=original_rate, target_sr=target_rate)
    return resampled.astype(np.float32).tolist()


def load_librispeech_stream(split: str, sample_rate: int, max_samples: int | None) -> Dataset:
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split=split,
        streaming=True,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate, decode=False))
    if max_samples:
        dataset = dataset.take(max_samples)
    records: dict[str, list[Any]] = {"audio": [], "text": [], "speaker_id": []}
    for sample in dataset:
        audio_info = sample["audio"]
        audio_array, original_rate = sf.read(io.BytesIO(audio_info["bytes"]))
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        resampled = resample_audio(audio_array.tolist(), original_rate, sample_rate)
        records["audio"].append(resampled)
        records["text"].append(sample["text"])
        records["speaker_id"].append(sample.get("speaker_id", -1))
    return Dataset.from_dict(records)


def prepare_features(batch: dict[str, Any], processor: Any, sample_rate: int) -> dict[str, Any]:
    audio = batch["audio"]
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    labels = processor.tokenizer(batch["text"], return_tensors="pt").input_ids
    return {
        "input_values": inputs.input_values[0],
        "attention_mask": inputs.attention_mask[0],
        "labels": labels[0],
    }


def to_tensor(values: Any) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values
    return torch.tensor(values)


def collate_features(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    input_values_list = [to_tensor(item["input_values"]).squeeze() for item in features]
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
        "input_values": torch.stack(input_batch),
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
        split = dataset.train_test_split(test_size=test_ratio, seed=seed)
        val_test = split["test"].train_test_split(test_size=0.5, seed=seed)
        return split["train"], val_test["train"], val_test["test"]

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
    return train, val, test


def load_dataset_split(config: DatasetConfig, sample_rate: int) -> Dataset:
    if config.dataset == "synthetic":
        return build_synthetic_dataset(sample_rate, config.max_seconds)
    if config.dataset == "librispeech_dummy":
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
    if config.dataset == "librispeech_clean":
        return load_librispeech_stream(config.split, sample_rate, config.max_samples)
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def normalize_audio(value: Any) -> list[float]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and "array" in value:
        return value["array"]
    if isinstance(value, dict) and "bytes" in value:
        raise ValueError("Audio bytes are not supported; provide arrays")
    raise ValueError("Unsupported audio format in manifest")


def prepare_dataset(dataset: Dataset, processor: Any) -> Dataset:
    sample_rate = processor.feature_extractor.sampling_rate
    return dataset.map(
        lambda batch: prepare_features(batch, processor, sample_rate),
        remove_columns=[col for col in dataset.column_names if col != "speaker_id"],
    )
