from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import io

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    set_seed,
)

DEFAULT_MODEL_ID = "UsefulSensors/moonshine-tiny"
DEFAULT_OUT_DIR = "outputs/poc"
DEFAULT_MAX_STEPS = 100
DEFAULT_EVAL_STEPS = 50
DEFAULT_SEED = 42
DEFAULT_DATASET = "librispeech_dummy"


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
    parser.add_argument("--dataset", choices=["librispeech_dummy", "synthetic"], default=DEFAULT_DATASET)
    parser.add_argument("--dataset-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mark_mps_fallback() -> bool:
    if not torch.backends.mps.is_available():
        return False
    return os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"


def load_processor(model_id: str) -> Any:
    token = os.environ.get("HF_TOKEN")
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = "<unk>"
    if processor.tokenizer.bos_token_id is None:
        processor.tokenizer.bos_token = "<s>"
    if processor.tokenizer.eos_token_id is None:
        processor.tokenizer.eos_token = "</s>"
    return processor


def generate_tone(duration: float, sample_rate: int, freq: float) -> list[float]:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (0.1 * np.sin(2 * math.pi * freq * t)).astype(np.float32).tolist()


def build_synthetic_dataset(sample_rate: int, max_seconds: float) -> Dataset:
    phrases = [
        ("beep one", 440.0),
        ("beep two", 554.0),
        ("beep three", 659.0),
        ("beep four", 880.0),
        ("beep five", 988.0),
        ("beep six", 523.0),
        ("beep seven", 740.0),
        ("beep eight", 830.0),
    ]
    records: dict[str, list[Any]] = {"audio": [], "text": []}
    for text, freq in phrases:
        duration = max(1.0, min(max_seconds, 2.0))
        audio = generate_tone(duration, sample_rate, freq)
        records["audio"].append(audio)
        records["text"].append(text)
    return Dataset.from_dict(records)


def resample_audio(audio: list[float], original_rate: int, target_rate: int) -> list[float]:
    if original_rate == target_rate:
        return audio
    resampled = librosa.resample(np.asarray(audio), orig_sr=original_rate, target_sr=target_rate)
    return resampled.astype(np.float32).tolist()


def load_librispeech_dummy(sample_rate: int, max_samples: int) -> Dataset:
    dataset = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation",
        streaming=True,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate, decode=False))
    if max_samples:
        dataset = dataset.take(max_samples)
    records: dict[str, list[Any]] = {"audio": [], "text": []}
    for sample in dataset:
        audio_info = sample["audio"]
        audio_array, original_rate = sf.read(io.BytesIO(audio_info["bytes"]))
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        resampled = resample_audio(audio_array.tolist(), original_rate, sample_rate)
        records["audio"].append(resampled)
        records["text"].append(sample["text"])
    return Dataset.from_dict(records)


def prepare_features(batch: dict[str, Any], processor: Any, sample_rate: int) -> dict[str, Any]:
    audio = batch["audio"]
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    labels = processor.tokenizer(batch["text"], return_tensors="pt").input_ids
    return {
        "input_values": inputs.input_values[0],
        "attention_mask": inputs.attention_mask[0],
        "labels": labels[0],
    }


def resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def setup_model(model_id: str, device: torch.device, lora_config: LoraConfig) -> Any:
    token = os.environ.get("HF_TOKEN")
    dtype = resolve_dtype(device)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=dtype, token=token)
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.config.use_cache = False
    return model


def find_lora_targets(model: Any) -> list[str]:
    candidates = [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "proj",
        "fc1",
        "fc2",
    ]
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    return [name for name in candidates if name in module_names]


def to_tensor(values: Any) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values
    return torch.tensor(values)


def collate_features(features: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    input_values_list = [to_tensor(item["input_values"]).squeeze() for item in features]
    attention_list = [to_tensor(item["attention_mask"]).squeeze() for item in features]
    labels_list = [to_tensor(item["labels"]) for item in features]
    input_lengths = [item.shape[-1] for item in input_values_list]
    max_input_len = max(input_lengths)
    input_batch = []
    attention_batch = []
    for input_values, attention_mask in zip(input_values_list, attention_list, strict=True):
        pad_len = max_input_len - input_values.shape[-1]
        if pad_len:
            input_values = F.pad(input_values, (0, pad_len))
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
        input_batch.append(input_values)
        attention_batch.append(attention_mask)

    label_lengths = [item.shape[-1] for item in labels_list]
    max_label_len = max(label_lengths)
    label_batch = []
    for labels in labels_list:
        pad_len = max_label_len - labels.shape[-1]
        if pad_len:
            labels = F.pad(labels, (0, pad_len), value=-100)
        label_batch.append(labels)

    return {
        "input_values": torch.stack(input_batch),
        "attention_mask": torch.stack(attention_batch),
        "labels": torch.stack(label_batch),
    }


def create_dataloader(dataset: Dataset, processor: Any, batch_size: int) -> DataLoader:
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda features: collate_features(features, pad_token_id),
    )


def unwrap_peft(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    if hasattr(model, "base_model"):
        return model.base_model
    return model


def decode_prediction(
    model: Any, processor: Any, batch: dict[str, Any], device: torch.device
) -> str:
    base_model = unwrap_peft(model)
    model_dtype = next(model.parameters()).dtype
    input_values = batch["input_values"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    if input_values.dim() == 1:
        input_values = input_values.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    input_values = input_values.to(model_dtype)
    with torch.no_grad():
        predicted_ids = base_model.generate(
            input_values=input_values, attention_mask=attention_mask
        )
    return processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]


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
    base_model = unwrap_peft(model)
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

    base_model.eval()
    eval_payload = {k: v.to(device) for k, v in eval_batch.items()}
    eval_payload["input_values"] = eval_payload["input_values"].to(model_dtype)
    with torch.no_grad():
        eval_outputs = base_model(**eval_payload)
    eval_loss = eval_outputs.loss.item()
    avg_train_loss = float(np.mean(losses)) if losses else 0.0
    return {"train_loss": avg_train_loss, "eval_loss": eval_loss}


def save_metrics(metrics: POCMetrics, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "poc_metrics.json"
    path.write_text(json.dumps(asdict(metrics), indent=2))


def run_poc(config: POCConfig) -> POCMetrics:
    set_seed(config.seed)
    device = choose_device()
    processor = load_processor(config.model_id)

    sample_rate = processor.feature_extractor.sampling_rate
    if config.dataset == "librispeech_dummy":
        dataset = load_librispeech_dummy(sample_rate, config.dataset_samples)
    else:
        dataset = build_synthetic_dataset(sample_rate, config.max_seconds)
    dataset = dataset.map(
        lambda batch: prepare_features(batch, processor, sample_rate),
        remove_columns=dataset.column_names,
    )

    split = dataset.train_test_split(test_size=0.2, seed=config.seed)
    train_loader = create_dataloader(split["train"], processor, config.batch_size)
    eval_batch = next(iter(create_dataloader(split["test"], processor, config.batch_size)))

    token = os.environ.get("HF_TOKEN")
    lora_targets = find_lora_targets(
        AutoModelForSpeechSeq2Seq.from_pretrained(config.model_id, token=token)
    )
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
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.bos_token_id = processor.tokenizer.bos_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.generation_config.bos_token_id = processor.tokenizer.bos_token_id
    model.generation_config.eos_token_id = processor.tokenizer.eos_token_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    baseline_text = decode_prediction(model, processor, eval_batch, device)

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

    tuned_text = decode_prediction(model, processor, eval_batch, device)

    output_dir = Path(config.output_dir)
    model.save_pretrained(output_dir / "lora_adapter")
    processor.save_pretrained(output_dir / "processor")

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
