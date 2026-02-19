"""Build a domain manifest JSONL file.

TODO: Address evaluation sample heterogeneity.
The current manifest generation logic results in a significant difficulty imbalance, 
specifically where the final ~50 samples (batches 150-200) are objectively easier 
than the initial samples. This causes the cumulative WER to "dive" at the end of 
the evaluation cycle rather than stabilizing. 

Future improvements should:
1. Increase the sample size (e.g., to 500+) to allow the metric to converge.
2. Implement a more robust shuffling mechanism or length-balancing to ensure 
   that "easy" and "hard" samples are distributed uniformly across the manifest.
3. Validate that the WER vs. Batch index plot flattens out before reporting final metrics.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

from datasets import Audio, IterableDataset, load_dataset
from soundfile import SoundFile

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManifestEntry:
    audio: list[float]
    text: str
    speaker_id: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the domain manifest builder.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Build a domain manifest JSONL file")
    parser.add_argument("--config", default="other")
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    return parser.parse_args()


def load_dataset_stream(config: str, split: str, seed: int, shuffle_buffer: int) -> IterableDataset:
    """Load a streaming LibriSpeech dataset.

    Args:
        config: LibriSpeech config name (e.g., clean, other).
        split: Dataset split name.

    Returns:
        Streaming dataset iterator.
    """
    dataset = load_dataset("librispeech_asr", config, split=split, streaming=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
    return dataset


def build_entries(dataset: IterableDataset, samples: int) -> list[ManifestEntry]:
    """Build manifest entries from a streaming dataset.

    Args:
        dataset: Streaming dataset iterator.
        samples: Number of samples to collect.

    Returns:
        Manifest entries.
    """
    entries: list[ManifestEntry] = []
    for sample in islice(dataset, samples):
        audio_info = sample["audio"]
        with SoundFile(io.BytesIO(audio_info["bytes"])) as sound:
            audio_array = sound.read(dtype="float32")
        entries.append(
            ManifestEntry(
                audio=audio_array.tolist(),
                text=sample["text"],
                speaker_id=sample.get("speaker_id", -1),
            )
        )
    if len(entries) < samples:
        raise ValueError(
            f"Requested {samples} samples but only loaded {len(entries)}"
        )
    return entries


def write_manifest(entries: list[ManifestEntry], output_path: Path) -> None:
    """Write manifest entries to disk.

    Args:
        entries: Manifest entries.
        output_path: Output JSONL file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(asdict(entry)) for entry in entries)
    output_path.write_text(payload)


def main() -> None:
    """Build a domain manifest with LibriSpeech "other" samples."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    dataset = load_dataset_stream(args.config, args.split, args.seed, args.shuffle_buffer)
    entries = build_entries(dataset, args.samples)
    output_path = Path(args.output)
    write_manifest(entries, output_path)
    LOGGER.info(
        "Manifest written | path=%s | samples=%s | config=%s | split=%s | seed=%s",
        output_path,
        len(entries),
        args.config,
        args.split,
        args.seed,
    )


if __name__ == "__main__":
    main()
