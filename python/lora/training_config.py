from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    model_id: str
    output_dir: str
    max_steps: int
    eval_steps: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    gradient_accumulation_steps: int
    seed: int
    dataset: str
    dataset_split: str
    dataset_samples: int
    max_seconds: float
    device: str | None


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
