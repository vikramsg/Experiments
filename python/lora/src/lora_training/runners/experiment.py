"""Unified experiment runner for baseline/tuned LoRA comparisons.

Output contract (under ``--output-dir``):
- ``experiment_metrics.json``: run-level metrics including baseline/tuned WER and loss.
- ``lora_adapter/``: adapter weights saved with ``save_pretrained``.
- ``processor/``: processor snapshot used for training/evaluation parity.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from transformers import AutoModelForSpeechSeq2Seq, get_linear_schedule_with_warmup, set_seed

from lora_data.data_loader import (
    DatasetConfig,
    create_dataloader,
    load_dataset_split,
    load_manifest_dataset,
    prepare_dataset,
    split_by_speaker,
)
from lora_training.evaluation import eval_loss, eval_wer, summarize_losses
from lora_training.logging_utils import get_logger, setup_logging
from lora_training.model_utils import (
    choose_device,
    configure_generation,
    find_lora_targets,
    load_processor,
    mark_mps_fallback,
    setup_model,
)
from lora_training.training_config import ExperimentConfig

DEFAULT_MODEL_ID = "UsefulSensors/moonshine-streaming-tiny"
DEFAULT_OUTPUT_DIR = "outputs/experiment"
DEFAULT_DATASET_PATH = "data/train_manifest_expanded.jsonl"
DEFAULT_MANIFEST_PATH = "data/heldout_manifest.jsonl"
DEFAULT_MAX_STEPS = 200
DEFAULT_EVAL_INTERVAL = 100
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_MAX_SECONDS = 20.0
DEFAULT_SEED = 42

LOGGER = get_logger(__name__)


@dataclass
class ExperimentMetrics:
    train_loss: float
    baseline_eval_loss: float
    baseline_wer: float
    tuned_eval_loss: float
    tuned_wer: float
    safety_baseline_wer: float | None
    safety_tuned_wer: float | None
    best_wer: float | None
    best_step: int | None
    interval_wer: float | None
    elapsed_seconds: float
    device: str
    used_mps_fallback: bool
    max_steps: int
    learning_rate: float
    eval_interval: int
    dataset_path: str
    manifest_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified LoRA experiment runner")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--safety-manifest-path",
        default=None,
        help="Optional manifest for regression guardrails.",
    )
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default=None)
    parser.add_argument(
        "--use-dora",
        action="store_true",
        help="Use Weight-Decomposed Low-Rank Adaptation (DoRA)",
    )
    parser.add_argument(
        "--init-lora-weights",
        default="gaussian",
        help="LoRA weight initialization (e.g., gaussian, pissa)",
    )
    parser.add_argument(
        "--wer-stop-threshold",
        type=float,
        default=None,
        help="Stop early if tuned WER exceeds baseline by this ratio.",
    )
    parser.add_argument(
        "--lora-module-filter",
        default=None,
        help="Substring filter for LoRA target modules (e.g., 'decoder').",
    )
    parser.add_argument(
        "--lora-targets",
        default=None,
        help="Comma-separated list of linear layer types to target (e.g., 'q_proj,v_proj').",
    )
    return parser.parse_args()


def _filter_manifest_by_duration(dataset: Any, sample_rate: int, max_seconds: float) -> Any:
    """Filter manifest items by duration.

    Args:
        dataset: Manifest dataset.
        sample_rate: Audio sample rate.
        max_seconds: Maximum duration in seconds.

    Returns:
        Filtered dataset.
    """
    duration_limit = int(sample_rate * max_seconds)
    return dataset.filter(lambda sample: len(sample["audio"]) <= duration_limit)


def _load_training_dataset(dataset_path: str, processor: Any, max_seconds: float, seed: int) -> Any:
    """Load training data from manifest or dataset.

    Args:
        dataset_path: Path to manifest or dataset identifier.
        processor: Model processor.
        max_seconds: Maximum audio duration.
        seed: Random seed.

    Returns:
        Prepared dataset for training.
    """
    match dataset_path:
        case path if path.startswith("db://") or path.endswith(".jsonl"):
            LOGGER.info("Loading manifest dataset | path=%s", dataset_path)
            dataset = load_manifest_dataset(dataset_path)
            LOGGER.info("Loaded manifest dataset | samples=%s", len(dataset))
            filtered = _filter_manifest_by_duration(
                dataset, processor.feature_extractor.sampling_rate, max_seconds
            )
            return prepare_dataset(filtered, processor)
        case "synthetic" | "librispeech_dummy" | "librispeech_clean":
            dataset_config = DatasetConfig(
                dataset=dataset_path,
                split="train",
                max_samples=None,
                max_seconds=max_seconds,
                seed=seed,
            )
            LOGGER.info("Loading dataset split | dataset=%s", dataset_path)
            dataset = load_dataset_split(dataset_config, processor.feature_extractor.sampling_rate)
            return prepare_dataset(dataset, processor)
        case invalid_path:
            raise ValueError(f"Unsupported dataset path or identifier: '{invalid_path}'")


def _build_dataloaders(
    dataset_path: str,
    processor: Any,
    batch_size: int,
    seed: int,
    max_seconds: float,
) -> tuple[Any, Any]:
    """Build training/validation dataloaders.

    Args:
        dataset_path: Path to manifest or dataset identifier.
        processor: Model processor.
        batch_size: Batch size.
        seed: Random seed.
        max_seconds: Maximum audio duration.

    Returns:
        Tuple of train and validation dataloaders.
    """
    dataset = _load_training_dataset(dataset_path, processor, max_seconds, seed)
    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("Need at least two samples to create train/val splits")
    test_size = max(1, int(dataset_size * 0.2))
    test_size = min(test_size, dataset_size - 1)
    use_speaker_split = False
    if "speaker_id" in dataset.column_names:
        unique_speakers = {speaker for speaker in dataset["speaker_id"]}
        use_speaker_split = len(unique_speakers) >= 3 and dataset_size >= 10
    if use_speaker_split:
        train_dataset, val_dataset, _ = split_by_speaker(
            dataset, test_ratio=0.1, val_ratio=0.1, seed=seed
        )
    else:
        split = dataset.train_test_split(test_size=test_size, seed=seed)
        train_dataset = split["train"]
        val_dataset = split["test"]
    train_loader = create_dataloader(train_dataset, batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size, shuffle=False)
    LOGGER.info(
        "Data split | train=%s | val=%s",
        len(train_dataset),
        len(val_dataset),
    )
    return train_loader, val_loader


def _load_eval_loader(
    manifest_path: str, processor: Any, batch_size: int, max_seconds: float
) -> Any:
    """Load evaluation dataloader from manifest.

    Args:
        manifest_path: Path to manifest JSONL or DB.
        processor: Model processor.
        batch_size: Batch size.
        max_seconds: Maximum audio duration.

    Returns:
        Evaluation dataloader.
    """
    dataset = load_manifest_dataset(manifest_path)
    dataset = _filter_manifest_by_duration(
        dataset, processor.feature_extractor.sampling_rate, max_seconds
    )
    prepared = prepare_dataset(dataset, processor)
    LOGGER.info("Eval manifest loaded | path=%s | samples=%s", manifest_path, len(prepared))
    return create_dataloader(prepared, batch_size, shuffle=False)


def _resolve_lora_targets(config: ExperimentConfig) -> list[str]:
    """Resolve LoRA target modules for a model.

    Args:
        config: Experiment configuration.

    Returns:
        List of module names to target with LoRA.
    """
    base_probe = AutoModelForSpeechSeq2Seq.from_pretrained(config.model_id)
    lora_targets = find_lora_targets(
        model=base_probe,
        module_filter=config.lora_module_filter,
        target_modules=config.lora_targets,
    )
    del base_probe
    if not lora_targets:
        lora_targets = ["q_proj", "v_proj"]
    LOGGER.info("LoRA targets | modules=%s", lora_targets)
    return lora_targets


def train_loop(
    model: Any,
    train_loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    max_steps: int,
    gradient_accumulation_steps: int,
    eval_interval: int,
    eval_loader: Any,
    processor: Any,
    baseline_wer: float,
    wer_stop_threshold: float | None,
    output_dir: Path | None = None,
    safety_loader: Any | None = None,
    exp_id: int | None = None,
    eval_ds_id: int | None = None,
) -> tuple[float, float | None, float | None, int | None, float | None]:
    """Train LoRA adapters and optionally run interval WER checks.

    Args:
        model: Adapter-wrapped model.
        train_loader: Training dataloader.
        optimizer: Optimizer for LoRA params.
        scheduler: LR scheduler.
        device: Torch device for training.
        max_steps: Training steps.
        gradient_accumulation_steps: Gradient accumulation steps.
        eval_interval: Steps between WER evaluations.
        eval_loader: Evaluation dataloader for WER checks.
        processor: Processor for decoding.
        baseline_wer: Baseline WER used for stop checks.
        wer_stop_threshold: Ratio threshold for early stop.
        output_dir: Directory to save best checkpoint.
        safety_loader: Evaluation dataloader for safety/guardrail checks.

    Returns:
        Tuple of (mean train loss, last interval WER, best WER, best step, last safety WER).
    """
    model_dtype = next(model.parameters()).dtype
    model.train()
    trainable_params = [
        param for name, param in model.named_parameters() if param.requires_grad and "lora" in name
    ]
    if not trainable_params:
        raise ValueError("No trainable LoRA parameters found")
    trainable_count = sum(param.numel() for param in trainable_params)
    LOGGER.info("Trainable LoRA params | count=%s", trainable_count)
    initial_weights = [param.detach().clone() for param in trainable_params]
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

    interval_wer: float | None = None
    safety_wer: float | None = None
    best_wer: float = float("inf")
    best_step: int | None = None
    stop_training = False
    while step < max_steps:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["input_values"] = batch["input_values"].to(model_dtype)
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            losses.append(loss.item() * gradient_accumulation_steps)
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
                update_step = (step + 1) // gradient_accumulation_steps
                if update_step == 1 or update_step % 10 == 0 or step + 1 == max_steps:
                    lr = optimizer.param_groups[0]["lr"]
                    LOGGER.info(
                        "Update %s/%s | step=%s | loss=%.4f | lr=%.2e",
                        update_step,
                        total_updates,
                        step + 1,
                        losses[-1],
                        lr,
                    )
            step += 1
            if eval_interval and step % eval_interval == 0:
                LOGGER.info("Evaluating DOMAIN manifest (Interval step=%s)", step)
                interval_wer = eval_wer(model, processor, eval_loader, device)
                LOGGER.info("Interval WER | step=%s | wer=%.4f", step, interval_wer)

                if safety_loader:
                    LOGGER.info("Evaluating HELDOUT manifest (Safety Interval step=%s)", step)
                    safety_wer = eval_wer(model, processor, safety_loader, device)
                    LOGGER.info("Interval Safety WER | step=%s | wer=%.4f", step, safety_wer)

                if exp_id is not None:
                    from db.client import DBClient
                    from db.models import RunMetric

                    with DBClient().session_scope() as session:
                        metric = RunMetric(
                            run_id=exp_id, step=step, key="interval_wer", value=interval_wer
                        )
                        session.add(metric)

                if interval_wer < best_wer:
                    best_wer = interval_wer
                    best_step = step
                    if output_dir:
                        best_path = output_dir / "lora_adapter_best"
                        model.save_pretrained(best_path)
                        LOGGER.info("New best WER! Saved adapter | path=%s", best_path)

                if wer_stop_threshold and interval_wer > baseline_wer * wer_stop_threshold:
                    LOGGER.warning(
                        "WER stop threshold exceeded | step=%s | wer=%.4f | baseline=%.4f",
                        step,
                        interval_wer,
                        baseline_wer,
                    )
                    stop_training = True
                    break
            if step >= max_steps:
                break
        if stop_training:
            break

    deltas = [
        float((param.detach() - initial).abs().sum().item())
        for param, initial in zip(trainable_params, initial_weights, strict=True)
    ]
    if sum(deltas) == 0.0:
        raise ValueError("LoRA weights did not change after updates")
    LOGGER.info("LoRA weight delta | value=%.6f", sum(deltas))
    LOGGER.info("Training loop complete | steps=%s", step)
    return (
        summarize_losses(losses),
        interval_wer,
        (best_wer if best_wer != float("inf") else None),
        best_step,
        safety_wer,
    )


def save_metrics(metrics: ExperimentMetrics, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiment_metrics.json"
    path.write_text(json.dumps(asdict(metrics), indent=2))


def run_experiment(
    config: ExperimentConfig, exp_id: int | None = None, eval_ds_id: int | None = None
) -> ExperimentMetrics:
    """Run a full experiment cycle with baseline and tuned evaluation.

    Args:
        config: Experiment configuration.
        exp_id: Database tracker experiment ID.
        eval_ds_id: Database tracker evaluation dataset ID.

    Returns:
        Experiment metrics summary.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_path=output_dir / "experiment.log")
    LOGGER.info("Starting experiment | config=%s", asdict(config))
    set_seed(config.seed)
    device = choose_device(config.device)
    LOGGER.info("Device selected | device=%s", device)
    processor = load_processor(config.model_id)
    LOGGER.info("Processor loaded | model_id=%s", config.model_id)

    train_loader, val_loader = _build_dataloaders(
        config.dataset_path,
        processor,
        config.batch_size,
        config.seed,
        max_seconds=config.max_seconds,
    )
    eval_loader = _load_eval_loader(
        config.manifest_path,
        processor,
        config.batch_size,
        max_seconds=config.max_seconds,
    )
    safety_loader = None
    if config.safety_manifest_path:
        safety_loader = _load_eval_loader(
            config.safety_manifest_path,
            processor,
            config.batch_size,
            max_seconds=config.max_seconds,
        )

    lora_targets = _resolve_lora_targets(config)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=lora_targets,
        use_dora=config.use_dora,
        init_lora_weights=config.init_lora_weights,
    )
    model = setup_model(config.model_id, device, lora_config)
    configure_generation(model, processor)
    LOGGER.info("Model ready | model_id=%s", config.model_id)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    total_updates = config.max_steps // config.gradient_accumulation_steps
    num_warmup_steps = int(total_updates * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_updates,
    )

    baseline_eval_loss = eval_loss(model, next(iter(val_loader)), device)

    LOGGER.info("Evaluating DOMAIN manifest (Baseline)")
    baseline_wer = eval_wer(model, processor, eval_loader, device)

    LOGGER.info(
        "Baseline metrics | eval_loss=%.4f | wer=%.4f",
        baseline_eval_loss,
        baseline_wer,
    )
    safety_baseline_wer = None
    if safety_loader:
        LOGGER.info("Evaluating HELDOUT manifest (Safety Baseline)")
        safety_baseline_wer = eval_wer(model, processor, safety_loader, device)
        LOGGER.info("Safety baseline WER | wer=%.4f", safety_baseline_wer)

    start = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loss, interval_wer, best_wer, best_step, safety_tuned_wer = train_loop(
        model,
        train_loader,
        optimizer,
        scheduler,
        device,
        max_steps=config.max_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_interval=config.eval_interval,
        eval_loader=eval_loader,
        processor=processor,
        baseline_wer=baseline_wer,
        wer_stop_threshold=config.wer_stop_threshold,
        output_dir=output_dir,
        safety_loader=safety_loader,
        exp_id=exp_id,
        eval_ds_id=eval_ds_id,
    )
    elapsed = time.time() - start
    LOGGER.info("Training finished | elapsed=%.2fs", elapsed)

    tuned_eval_loss = eval_loss(model, next(iter(val_loader)), device)
    LOGGER.info("Evaluating DOMAIN manifest (Final Tuned)")
    tuned_wer = eval_wer(model, processor, eval_loader, device)
    LOGGER.info(
        "Final tuned metrics | eval_loss=%.4f | wer=%.4f",
        tuned_eval_loss,
        tuned_wer,
    )

    if safety_loader:
        LOGGER.info("Evaluating HELDOUT manifest (Final Safety Tuned)")
        safety_tuned_wer = eval_wer(model, processor, safety_loader, device)
        LOGGER.info("Safety final tuned WER | wer=%.4f", safety_tuned_wer)

    if best_wer is not None:
        LOGGER.info("Best metrics during training | wer=%.4f | step=%s", best_wer, best_step)

    if (
        config.wer_stop_threshold is not None
        and tuned_wer > baseline_wer * config.wer_stop_threshold
    ):
        LOGGER.warning(
            "WER regression threshold exceeded | tuned=%.4f | baseline=%.4f | ratio=%.2f",
            tuned_wer,
            baseline_wer,
            config.wer_stop_threshold,
        )

    model.save_pretrained(output_dir / "lora_adapter")
    processor.save_pretrained(output_dir / "processor")
    LOGGER.info("Artifacts saved | output_dir=%s", output_dir)

    return ExperimentMetrics(
        train_loss=train_loss,
        baseline_eval_loss=baseline_eval_loss,
        baseline_wer=baseline_wer,
        tuned_eval_loss=tuned_eval_loss,
        tuned_wer=tuned_wer,
        safety_baseline_wer=safety_baseline_wer,
        safety_tuned_wer=safety_tuned_wer,
        best_wer=best_wer,
        best_step=best_step,
        interval_wer=interval_wer,
        elapsed_seconds=elapsed,
        device=str(device),
        used_mps_fallback=mark_mps_fallback(),
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        eval_interval=config.eval_interval,
        dataset_path=config.dataset_path,
        manifest_path=config.manifest_path,
    )


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        dataset_path=args.dataset_path,
        manifest_path=args.manifest_path,
        safety_manifest_path=args.safety_manifest_path,
        max_seconds=args.max_seconds,
        device=args.device,
        wer_stop_threshold=args.wer_stop_threshold,
        use_dora=args.use_dora,
        init_lora_weights=args.init_lora_weights,
        lora_module_filter=args.lora_module_filter,
        lora_targets=args.lora_targets,
    )

    from db.client import DBClient
    from db.models import Dataset, Run, RunDataset, RunParam

    client = DBClient()
    client.init_db()

    with client.session_scope() as session:
        train_ds_id = None
        match config.dataset_path:
            case path if path.startswith("db://"):
                ds_name = path[5:]
                ds = session.query(Dataset).filter_by(name=ds_name).first()
                train_ds_id = ds.id if ds else None
            case path if path.endswith(".jsonl") or path in ("synthetic", "librispeech_dummy", "librispeech_clean"):
                pass
            case invalid_path:
                raise ValueError(f"Invalid dataset path format: {invalid_path}")

        eval_ds_id = None
        match config.manifest_path:
            case path if path.startswith("db://"):
                ds_name = path[5:]
                ds = session.query(Dataset).filter_by(name=ds_name).first()
                eval_ds_id = ds.id if ds else None
            case path if path.endswith(".jsonl"):
                pass
            case invalid_path:
                raise ValueError(f"Invalid eval manifest path format: {invalid_path}")

        safety_ds_id = None
        if config.safety_manifest_path:
            match config.safety_manifest_path:
                case path if path.startswith("db://"):
                    ds_name = path[5:]
                    ds = session.query(Dataset).filter_by(name=ds_name).first()
                    safety_ds_id = ds.id if ds else None
                case path if path.endswith(".jsonl"):
                    pass
                case invalid_path:
                    raise ValueError(f"Invalid safety manifest path format: {invalid_path}")

        exp_name = Path(config.output_dir).name
        log_file = str(Path(config.output_dir) / "experiment.log")
        raw_out = str(Path(config.output_dir) / "nohup.out")

        run = session.query(Run).filter_by(name=exp_name).first()
        if not run:
            run = Run(
                name=exp_name,
                run_type="TRAINING",
                status="RUNNING",
                log_file_path=log_file,
                raw_stdout_path=raw_out,
            )
            session.add(run)
            session.flush()
        else:
            run.status = "RUNNING"
            run.log_file_path = log_file
            run.raw_stdout_path = raw_out
            session.query(RunParam).filter_by(run_id=run.id).delete()
            session.query(RunDataset).filter_by(run_id=run.id).delete()
            session.flush()

        for key, value in asdict(config).items():
            session.add(RunParam(run_id=run.id, key=key, value=str(value)))

        if train_ds_id:
            session.add(RunDataset(run_id=run.id, dataset_id=train_ds_id, usage="TRAIN"))
        if eval_ds_id:
            session.add(RunDataset(run_id=run.id, dataset_id=eval_ds_id, usage="EVAL"))
        if safety_ds_id:
            session.add(RunDataset(run_id=run.id, dataset_id=safety_ds_id, usage="SAFETY"))

        exp_id = run.id

    try:
        metrics = run_experiment(config, exp_id=exp_id, eval_ds_id=eval_ds_id)
        save_metrics(metrics, Path(config.output_dir))
        with client.session_scope() as session:
            run = session.query(Run).get(exp_id)
            if run:
                run.status = "COMPLETED"
                run.artifacts_dir = str(Path(config.output_dir) / "lora_adapter")
    except Exception as e:
        import traceback

        with client.session_scope() as session:
            run = session.query(Run).get(exp_id)
            if run:
                run.status = "FAILED"
                run.error_traceback = traceback.format_exc()
        raise e


if __name__ == "__main__":
    main()
