"""Build a JSONL manifest of audio arrays + transcripts for STT checks.

Usage:
    uv run python -m lora.scripts.build_manifest --split test --samples 3 --output data/heldout_manifest.jsonl

Flags:
    --split     LibriSpeech split name (e.g. test, validation, train.100)
    --samples   Number of samples to include in the manifest
    --output    Output JSONL file path
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import Audio, load_dataset
from soundfile import SoundFile


@dataclass
class ManifestEntry:
    audio: list[float]
    text: str
    speaker_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a JSONL manifest for STT checks")
    parser.add_argument("--split", default="test.clean")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split=args.split,
        streaming=True,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for sample in dataset.take(args.samples):
        audio_info = sample["audio"]
        with SoundFile(io.BytesIO(audio_info["bytes"])) as sound:
            audio_array = sound.read(dtype="float32")
        entry = ManifestEntry(
            audio=audio_array.tolist(),
            text=sample["text"],
            speaker_id=sample.get("speaker_id", -1),
        )
        entries.append(json.dumps(asdict(entry)))
    output_path.write_text("\n".join(entries))


if __name__ == "__main__":
    main()
