from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lora_data.manifest_diagnostics import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_THRESHOLDS,
    ManifestProfile,
    SplitProfile,
    profile_manifest,
    split_manifest,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for manifest diagnostics.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Manifest diagnostics")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def log_manifest_profiles(profiles: list[ManifestProfile]) -> None:
    """Log manifest duration summaries.

    Args:
        profiles: Manifest profile entries.
    """
    for profile in profiles:
        LOGGER.info(
            "Duration profile | path=%s | samples=%s | mean=%.2fs | counts=%s",
            profile.path,
            profile.samples,
            profile.mean_seconds,
            profile.threshold_counts,
        )


def log_split_profiles(profiles: list[SplitProfile]) -> None:
    """Log train/val split summaries.

    Args:
        profiles: Split profile entries.
    """
    for profile in profiles:
        LOGGER.info(
            "Split profile | path=%s | max_seconds=%s | filtered=%s | train=%s | val=%s | split=%s",
            profile.path,
            profile.max_seconds,
            profile.filtered_samples,
            profile.train_samples,
            profile.val_samples,
            profile.split_strategy,
        )


def main() -> None:
    """Run manifest diagnostics from the CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    thresholds = args.thresholds or list(DEFAULT_THRESHOLDS)
    manifest_paths = [Path(value) for value in args.manifests]
    profiles = [profile_manifest(path, thresholds, args.sample_rate) for path in manifest_paths]
    splits = [
        split_manifest(path, args.max_seconds, args.sample_rate, args.seed)
        for path in manifest_paths
    ]
    log_manifest_profiles(profiles)
    log_split_profiles(splits)


if __name__ == "__main__":
    main()
