from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, Sequence, TypedDict, cast

from lora.data_loader import load_manifest_dataset, split_by_speaker

DEFAULT_THRESHOLDS: tuple[float, ...] = (5.0, 8.0, 10.0, 15.0, 20.0)
DEFAULT_SAMPLE_RATE = 16000


class ManifestEntry(TypedDict):
    audio: list[float]
    text: str
    speaker_id: NotRequired[int | str]


@dataclass(frozen=True)
class ManifestProfile:
    path: Path
    samples: int
    mean_seconds: float
    threshold_counts: dict[float, int]


@dataclass(frozen=True)
class SplitProfile:
    path: Path
    filtered_samples: int
    train_samples: int
    val_samples: int
    max_seconds: float
    split_strategy: str


def load_manifest_entries(path: Path) -> list[ManifestEntry]:
    """Load JSONL manifest entries from disk.

    Args:
        path: Path to the manifest JSONL file.

    Returns:
        Parsed manifest entries.
    """
    entries: list[ManifestEntry] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload: dict[str, object] = json.loads(line)
        entries.append(cast(ManifestEntry, payload))
    if not entries:
        raise ValueError(f"Manifest is empty: {path}")
    return entries


def profile_manifest(
    path: Path, thresholds: Sequence[float], sample_rate: int
) -> ManifestProfile:
    """Profile audio duration distribution for a manifest.

    Args:
        path: Manifest JSONL path.
        thresholds: Duration thresholds in seconds.
        sample_rate: Audio sample rate in Hz.

    Returns:
        ManifestProfile with counts per threshold.
    """
    entries = load_manifest_entries(path)
    durations = [len(entry["audio"]) / sample_rate for entry in entries]
    mean_seconds = statistics.fmean(durations)
    threshold_counts = {
        threshold: sum(duration <= threshold for duration in durations)
        for threshold in thresholds
    }
    return ManifestProfile(
        path=path,
        samples=len(entries),
        mean_seconds=mean_seconds,
        threshold_counts=threshold_counts,
    )


def split_manifest(
    path: Path, max_seconds: float, sample_rate: int, seed: int
) -> SplitProfile:
    """Compute train/val split sizes after duration filtering.

    Args:
        path: Manifest JSONL path.
        max_seconds: Maximum duration allowed in seconds.
        sample_rate: Audio sample rate in Hz.
        seed: Random seed for splits.

    Returns:
        SplitProfile describing the filtered dataset sizes.
    """
    dataset = load_manifest_dataset(path)
    duration_limit = int(sample_rate * max_seconds)
    filtered = dataset.filter(lambda sample: len(sample["audio"]) <= duration_limit)
    filtered_size = len(filtered)
    if filtered_size < 2:
        raise ValueError("Need at least two samples to create train/val splits")
    test_size = max(1, int(filtered_size * 0.2))
    test_size = min(test_size, filtered_size - 1)
    use_speaker_split = False
    if "speaker_id" in filtered.column_names:
        unique_speakers = {speaker for speaker in filtered["speaker_id"]}
        use_speaker_split = len(unique_speakers) >= 3 and filtered_size >= 10
    if use_speaker_split:
        train_dataset, val_dataset, _ = split_by_speaker(
            filtered, test_ratio=0.1, val_ratio=0.1, seed=seed
        )
        split_strategy = "speaker"
    else:
        split = filtered.train_test_split(test_size=test_size, seed=seed)
        train_dataset = split["train"]
        val_dataset = split["test"]
        split_strategy = "random"
    return SplitProfile(
        path=path,
        filtered_samples=filtered_size,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
        max_seconds=max_seconds,
        split_strategy=split_strategy,
    )
