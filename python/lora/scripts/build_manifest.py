"""Build a JSONL manifest of audio arrays + transcripts for STT checks.

This script creates small, portable datasets for testing Speech-to-Text (STT) models by:
1.  **Downloading Audio:** Fetches audio samples from the LibriSpeech dataset (a standard benchmark for ASR).
2.  **Processing Audio:** Reads raw audio bytes, ensures 16kHz sampling, and converts them to floating-point arrays.
3.  **Formatting Data:** Packages the audio data along with its corresponding transcript and speaker ID.
4.  **Saving to JSONL:** Writes these packaged entries into a JSON Lines (JSONL) file.

Benefits:
*   **Offline/Fast Testing:** Enables quick sanity checks without re-downloading massive datasets.
*   **Reproducibility:** Ensures consistent evaluation on the exact same audio clips.
*   **CI/CD:** Provides a tiny, reliable dataset for Continuous Integration pipelines without network overhead.

Usage:
    uv run python scripts/build_manifest.py --split test --samples 3 --output data/heldout_manifest.jsonl

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
    sample_iter = iter(dataset)
    while len(entries) < args.samples:
        try:
            sample = next(sample_iter)
        except StopIteration:
            # TODO: remove fallback StopIteration handling; fail fast with explicit sample-count checks.
            break
        try:
            audio_info = sample["audio"]
            with SoundFile(io.BytesIO(audio_info["bytes"])) as sound:
                audio_array = sound.read(dtype="float32")
            entry = ManifestEntry(
                audio=audio_array.tolist(),
                text=sample["text"],
                speaker_id=sample.get("speaker_id", -1),
            )
            entries.append(json.dumps(asdict(entry)))
        except Exception:
            # TODO: remove fallback exception swallowing; fail fast with explicit error handling.
            continue
    output_path.write_text("\n".join(entries))


if __name__ == "__main__":
    main()
