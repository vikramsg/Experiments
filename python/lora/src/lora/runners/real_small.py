from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from torch.utils.data import DataLoader
from transformers import AutoModelForSpeechSeq2Seq, set_seed

from lora.data_loader import (
    DatasetConfig,
    create_dataloader,
    load_dataset_split,
    prepare_dataset,
    split_by_speaker,
)
from lora.evaluation import eval_loss, eval_wer, summarize_losses
from lora.model_utils import (
    choose_device,
    configure_generation,
    find_lora_targets,
    load_processor,
    mark_mps_fallback,
    setup_model,
)
from lora.training_config import RealRunConfig

DEFAULT_MODEL_ID = "UsefulSensors/moonshine-tiny"
DEFAULT_OUTPUT_DIR = "outputs/real_small"


@dataclass
class RealRunMetrics:
    train_loss: float
    baseline_eval_loss: float
    baseline_wer: float
    tuned_eval_loss: float
    tuned_wer: float
    elapsed_seconds: float
    device: str
    used_mps_fallback: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small real-world LoRA run")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-samples", type=int, default=400)
    parser.add_argument("--train-split", default="train.100")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default=None)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--wer-batches", type=int, default=12)
    return parser.parse_args()


def train_loop(
    model: Any,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int,
    gradient_accumulation_steps: int,
) -> float:
    base_model = model.get_base_model()
    model_dtype = next(model.parameters()).dtype
    base_model.train()
    losses: list[float] = []
    step = 0
    optimizer.zero_grad()

    while step < max_steps:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["input_values"] = batch["input_values"].to(model_dtype)
            outputs = base_model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            losses.append(loss.item() * gradient_accumulation_steps)
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            step += 1
            if step >= max_steps:
                break

    return summarize_losses(losses)


def save_metrics(metrics: RealRunMetrics, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "real_metrics.json"
    path.write_text(json.dumps(asdict(metrics), indent=2))


def run_real(config: RealRunConfig) -> RealRunMetrics:
    set_seed(config.seed)
    device = choose_device(config.device)
    processor = load_processor(config.model_id)
    sample_rate = processor.feature_extractor.sampling_rate

    dataset_config = DatasetConfig(
        dataset="librispeech_clean",
        split=config.train_split,
        max_samples=config.dataset_samples,
        max_seconds=config.max_seconds,
        seed=config.seed,
    )
    raw_dataset = load_dataset_split(dataset_config, sample_rate)
    train_raw, val_raw, test_raw = split_by_speaker(
        raw_dataset, test_ratio=0.1, val_ratio=0.1, seed=config.seed
    )
    train_dataset = prepare_dataset(train_raw, processor)
    val_dataset = prepare_dataset(val_raw, processor)
    test_dataset = prepare_dataset(test_raw, processor)

    train_loader = create_dataloader(train_dataset, config.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, config.batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, config.batch_size, shuffle=False)

    base_probe = AutoModelForSpeechSeq2Seq.from_pretrained(config.model_id)
    lora_targets = find_lora_targets(model=base_probe)
    del base_probe
    if not lora_targets:
        lora_targets = ["q_proj", "v_proj"]
    if not lora_targets:
        lora_targets = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=lora_targets,
        task_type="SEQ_2_SEQ_LM",
    )
    model = setup_model(config.model_id, device, lora_config)
    configure_generation(model, processor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    baseline_eval_loss = eval_loss(model, next(iter(val_loader)), device)
    baseline_wer = eval_wer(model, processor, test_loader, device, max_batches=config.wer_batches)

    start = time.time()
    train_loss = train_loop(
        model,
        train_loader,
        optimizer,
        device,
        max_steps=config.max_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    elapsed = time.time() - start

    tuned_eval_loss = eval_loss(model, next(iter(val_loader)), device)
    tuned_wer = eval_wer(model, processor, test_loader, device, max_batches=config.wer_batches)

    output_dir = Path(config.output_dir)
    model.save_pretrained(output_dir / "lora_adapter")
    processor.save_pretrained(output_dir / "processor")

    return RealRunMetrics(
        train_loss=train_loss,
        baseline_eval_loss=baseline_eval_loss,
        baseline_wer=baseline_wer,
        tuned_eval_loss=tuned_eval_loss,
        tuned_wer=tuned_wer,
        elapsed_seconds=elapsed,
        device=str(device),
        used_mps_fallback=mark_mps_fallback(),
    )


def main() -> None:
    args = parse_args()
    config = RealRunConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        dataset_samples=args.dataset_samples,
        train_split=args.train_split,
        device=args.device,
        max_seconds=args.max_seconds,
        wer_batches=args.wer_batches,
    )
    metrics = run_real(config)
    save_metrics(metrics, Path(config.output_dir))
    print(json.dumps(asdict(metrics), indent=2))


if __name__ == "__main__":
    main()
