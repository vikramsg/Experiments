"""Typed runtime configuration models used by training runners."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str
    output_dir: str
    max_steps: int
    eval_interval: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    gradient_accumulation_steps: int
    seed: int
    dataset_path: str
    manifest_path: str
    max_seconds: float
    device: str | None
    wer_stop_threshold: float | None


@dataclass
class RealRunConfig:
    model_id: str
    output_dir: str
    max_steps: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    gradient_accumulation_steps: int
    seed: int
    dataset_samples: int
    train_split: str
    device: str | None
    max_seconds: float
    wer_batches: int
    manifest_path: str | None
