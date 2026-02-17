from __future__ import annotations

import os
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def choose_device(preferred: str | None = None) -> torch.device:
    if preferred:
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "cpu":
            return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mark_mps_fallback() -> bool:
    if not torch.backends.mps.is_available():
        return False
    return os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"


def resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def load_processor(model_id: str, processor_dir: str | None = None) -> Any:
    token = os.environ.get("HF_TOKEN")
    source = processor_dir or model_id
    processor = AutoProcessor.from_pretrained(source, token=token)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = "<unk>"
    if processor.tokenizer.bos_token_id is None:
        processor.tokenizer.bos_token = "<s>"
    if processor.tokenizer.eos_token_id is None:
        processor.tokenizer.eos_token = "</s>"
    return processor


def setup_model(model_id: str, device: torch.device, lora_config: LoraConfig) -> Any:
    token = os.environ.get("HF_TOKEN")
    dtype = resolve_dtype(device)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=dtype, token=token)
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.config.use_cache = False
    return model


def configure_generation(model: Any, processor: Any) -> None:
    pad_token_id = processor.tokenizer.pad_token_id
    bos_token_id = processor.tokenizer.bos_token_id
    eos_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = pad_token_id
    model.config.decoder_start_token_id = bos_token_id
    model.config.bos_token_id = bos_token_id
    model.config.eos_token_id = eos_token_id
    model.generation_config.pad_token_id = pad_token_id
    model.generation_config.decoder_start_token_id = bos_token_id
    model.generation_config.bos_token_id = bos_token_id
    model.generation_config.eos_token_id = eos_token_id


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


def unwrap_peft(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    if hasattr(model, "base_model"):
        return model.base_model
    return model
