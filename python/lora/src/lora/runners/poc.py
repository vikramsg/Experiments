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
    build_synthetic_dataset,
    create_dataloader,
    load_dataset_split,
    prepare_dataset,
)
from lora.evaluation import decode_prediction, eval_loss, summarize_losses
from lora.logging_utils import get_logger, setup_logging
from lora.model_utils import (
    choose_device,
    configure_generation,
    find_lora_targets,
    load_processor,
    mark_mps_fallback,
    setup_model,
)

DEFAULT_MODEL_ID = "UsefulSensors/moonshine-tiny"
DEFAULT_OUT_DIR = "outputs/poc"
DEFAULT_MAX_STEPS = 100
DEFAULT_EVAL_STEPS = 50
DEFAULT_SEED = 42
DEFAULT_DATASET = "librispeech_dummy"

LOGGER = get_logger(__name__)


@dataclass
class POCConfig:
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
    max_seconds: float
    use_mps: bool
    seed: int
    dataset: str
    dataset_samples: int


@dataclass
class POCMetrics:
    train_loss: float
    eval_loss: float
    baseline_text: str
    tuned_text: str
    elapsed_seconds: float
    device: str
    used_mps_fallback: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Moonshine LoRA POC runner")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument(
        "--dataset",
        choices=["librispeech_dummy", "synthetic"],
        default=DEFAULT_DATASET,
    )
    parser.add_argument("--dataset-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def train_loop(
    model: Any,
    train_loader: DataLoader,
    eval_batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int,
    gradient_accumulation_steps: int,
    eval_steps: int,
) -> dict[str, float]:
    _ = eval_steps
    base_model = model.get_base_model()
    model_dtype = next(model.parameters()).dtype
    base_model.train()
    trainable_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and "lora" in name
    ]
    if not trainable_params:
        raise ValueError("No trainable LoRA parameters found")
    trainable_count = sum(param.numel() for param in trainable_params)
    LOGGER.info("Trainable LoRA params | count=%s", trainable_count)
    initial_weights = [param.detach().clone() for param in trainable_params]
    first_grad_logged = False
    delta_logged = False
    losses: list[float] = []
    step = 0
    optimizer.zero_grad()
    total_updates = max_steps // gradient_accumulation_steps
    LOGGER.info(
        "Training start | steps=%s updates=%s grad_accum=%s",
        max_steps,
        total_updates,
        gradient_accumulation_steps,
    )

    while step < max_steps:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["input_values"] = batch["input_values"].to(model_dtype)
            outputs = base_model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            if not first_grad_logged:
                # TODO: remove inline grad fallback; make explicit and fail fast.
                grad_values = [
                    float(param.grad.detach().norm().item()) if param.grad is not None else 0.0
                    for param in trainable_params
                ]
                grad_norm = float(torch.tensor(grad_values).norm().item())
                if grad_norm == 0.0:
                    raise ValueError("LoRA gradients are zero on first backward pass")
                LOGGER.info("LoRA grad norm | value=%.6f", grad_norm)
                first_grad_logged = True
            losses.append(loss.item() * gradient_accumulation_steps)
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                update_step = (step + 1) // gradient_accumulation_steps
                if not delta_logged and update_step >= 3:
                    deltas = [
                        float((param.detach() - initial).abs().sum().item())
                        for param, initial in zip(
                            trainable_params, initial_weights, strict=True
                        )
                    ]
                    if sum(deltas) == 0.0:
                        raise ValueError("LoRA weights did not change after updates")
                    LOGGER.info("LoRA weight delta | value=%.6f", sum(deltas))
                    delta_logged = True
                if update_step == 1 or update_step % 5 == 0 or step + 1 == max_steps:
                    LOGGER.info(
                        "Update %s/%s | step=%s | loss=%.4f",
                        update_step,
                        total_updates,
                        step + 1,
                        losses[-1],
                    )
            step += 1
            if step >= max_steps:
                break

    LOGGER.info("Training loop complete | steps=%s", step)
    eval_loss_value = eval_loss(model, eval_batch, device)
    LOGGER.info("Eval loss complete | loss=%.4f", eval_loss_value)
    avg_train_loss = summarize_losses(losses)
    return {"train_loss": avg_train_loss, "eval_loss": eval_loss_value}


def save_metrics(metrics: POCMetrics, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "poc_metrics.json"
    path.write_text(json.dumps(asdict(metrics), indent=2))


def run_poc(config: POCConfig) -> POCMetrics:
    setup_logging()
    set_seed(config.seed)
    device = choose_device()
    LOGGER.info("Device selected | device=%s", device)
    processor = load_processor(config.model_id)
    LOGGER.info("Processor loaded | model_id=%s", config.model_id)

    sample_rate = processor.feature_extractor.sampling_rate
    LOGGER.info("Loading dataset | name=%s | sample_rate=%s", config.dataset, sample_rate)
    dataset_config = DatasetConfig(
        dataset=config.dataset,
        split="validation",
        max_samples=config.dataset_samples,
        max_seconds=config.max_seconds,
        seed=config.seed,
    )
    if config.dataset == "synthetic":
        dataset = build_synthetic_dataset(sample_rate, config.max_seconds)
    else:
        # TODO: remove fallback else; make dataset choice explicit and fail fast.
        dataset = load_dataset_split(dataset_config, sample_rate)
    dataset = prepare_dataset(dataset, processor)
    LOGGER.info("Prepared dataset | samples=%s", len(dataset))

    split = dataset.train_test_split(test_size=0.2, seed=config.seed)
    train_loader = create_dataloader(split["train"], config.batch_size, shuffle=True)
    eval_batch = next(iter(create_dataloader(split["test"], config.batch_size, shuffle=False)))
    LOGGER.info(
        "Data split | train=%s | eval=%s",
        len(split["train"]),
        len(split["test"]),
    )

    lora_targets = find_lora_targets(AutoModelForSpeechSeq2Seq.from_pretrained(config.model_id))
    if not lora_targets:
        lora_targets = ["q_proj", "v_proj"]
    LOGGER.info("LoRA targets | modules=%s", lora_targets)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=lora_targets,
    )

    model = setup_model(config.model_id, device, lora_config)
    configure_generation(model, processor)
    LOGGER.info("Model ready | model_id=%s", config.model_id)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    baseline_text = decode_prediction(model, processor, eval_batch, device)
    LOGGER.info("Baseline decode complete")

    start = time.time()
    losses = train_loop(
        model=model,
        train_loader=train_loader,
        eval_batch=eval_batch,
        optimizer=optimizer,
        device=device,
        max_steps=config.max_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_steps=config.eval_steps,
    )
    elapsed = time.time() - start
    LOGGER.info("Training finished | elapsed=%.2fs", elapsed)

    tuned_text = decode_prediction(model, processor, eval_batch, device)
    LOGGER.info("Tuned decode complete")

    output_dir = Path(config.output_dir)
    model.save_pretrained(output_dir / "lora_adapter")
    processor.save_pretrained(output_dir / "processor")
    LOGGER.info("Artifacts saved | output_dir=%s", output_dir)

    return POCMetrics(
        train_loss=losses["train_loss"],
        eval_loss=losses["eval_loss"],
        baseline_text=baseline_text,
        tuned_text=tuned_text,
        elapsed_seconds=elapsed,
        device=str(device),
        used_mps_fallback=mark_mps_fallback(),
    )


def main() -> None:
    args = parse_args()
    config = POCConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seconds=args.max_seconds,
        use_mps=True,
        seed=args.seed,
        dataset=args.dataset,
        dataset_samples=args.dataset_samples,
    )
    metrics = run_poc(config)
    save_metrics(metrics, Path(config.output_dir))
    print(json.dumps(asdict(metrics), indent=2))


if __name__ == "__main__":
    main()
