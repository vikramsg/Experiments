from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger(__name__)


def iter_manifest_entries(path: Path) -> Iterable[dict[str, object]]:
    """Yield parsed manifest entries from a JSONL file.

    Args:
        path: Path to a JSONL manifest file.

    Yields:
        Parsed manifest entries.
    """
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        yield json.loads(line)


def merge_manifests(input_paths: list[Path], output_path: Path) -> int:
    """Merge manifest files into a single JSONL output.

    Args:
        input_paths: Manifest JSONL files to merge.
        output_path: Output path for the merged manifest.

    Returns:
        Number of entries written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_entries = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for input_path in input_paths:
            for entry in iter_manifest_entries(input_path):
                handle.write(json.dumps(entry))
                handle.write("\n")
                total_entries += 1
    return total_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge JSONL manifests")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    input_paths = [Path(value) for value in args.inputs]
    output_path = Path(args.output)
    total_entries = merge_manifests(input_paths, output_path)
    LOGGER.info("Merged manifests | inputs=%s | output=%s | entries=%s", input_paths, output_path, total_entries)


if __name__ == "__main__":
    main()
