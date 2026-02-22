"""Utility functions for dataset manifests."""

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def split_manifest(input_manifest: str | Path, train_ratio: float = 0.8, seed: int = 42):
    """Split a master JSONL manifest into train and eval sets."""
    in_path = Path(input_manifest)
    if not in_path.exists():
        logger.error(f"Manifest not found: {in_path}")
        return

    entries = []
    with open(in_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        logger.warning(f"Manifest {in_path} is empty.")
        return

    random.seed(seed)
    random.shuffle(entries)

    split_idx = int(len(entries) * train_ratio)
    train_entries = entries[:split_idx]
    eval_entries = entries[split_idx:]

    base_name = in_path.stem.replace("_all", "")
    train_path = in_path.parent / f"{base_name}_train.jsonl"
    eval_path = in_path.parent / f"{base_name}_eval.jsonl"

    def write_jsonl(path, data):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item))
                f.write("\n")

    write_jsonl(train_path, train_entries)
    write_jsonl(eval_path, eval_entries)

    logger.info("Split complete:")
    logger.info(f"  Total: {len(entries)}")
    logger.info(f"  Train: {len(train_entries)} -> {train_path}")
    logger.info(f"  Eval:  {len(eval_entries)} -> {eval_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/my_voice_all.jsonl")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    # Configure basic logger for CLI usage
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    split_manifest(args.manifest, args.train_ratio)
