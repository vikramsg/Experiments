"""Model and processor setup helpers shared by training and inference flows."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoProcessor

CTC_MODEL_TYPES = {
    "hubert",
    "unispeech",
    "unispeech_sat",
    "wav2vec2",
    "wav2vec2_conformer",
    "wavlm",
}


def choose_device(preferred: str | None = None) -> torch.device:
    if preferred:
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "cpu":
            return torch.device("cpu")
        # TODO: remove fallback device selection; fail fast on unsupported preference.
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
    # TODO: remove fallback dtype; make device dtype explicit and fail fast.
    return torch.float32


def normalize_audio_rms(audio_data: Sequence[float], target_rms: float = 0.075) -> list[float]:
    """Normalize audio amplitude to a target RMS value.

    Args:
        audio_data: Audio samples in waveform order.
        target_rms: Desired RMS value after normalization.

    Returns:
        RMS-normalized audio samples as a list of floats.
    """
    tensor = torch.tensor(audio_data, dtype=torch.float32)
    rms = torch.sqrt(torch.mean(tensor**2)).item()
    if rms > 0.001:
        scale_factor = target_rms / rms
        normalized = tensor * scale_factor
        return torch.clamp(normalized, -1.0, 1.0).tolist()
    # TODO: remove fallback audio normalization; require explicit minimum RMS handling.
    return tensor.tolist()


def is_ctc_config(config: Any) -> bool:
    return getattr(config, "model_type", None) in CTC_MODEL_TYPES


def is_ctc_model(model_id: str) -> bool:
    config = AutoConfig.from_pretrained(model_id)
    return is_ctc_config(config)


def _safe_set_token(tokenizer: Any, token_id: int | None, attr: str) -> None:
    if token_id is None:
        # TODO: remove fallback token checks; fail fast on missing token IDs.
        return
    token = tokenizer.convert_ids_to_tokens(token_id)
    if token is None:
        # TODO: remove fallback token checks; fail fast on missing tokens.
        return
    setattr(tokenizer, attr, token)


def _register_special_tokens(tokenizer: Any) -> None:
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if not callable(get_added_vocab):
        # TODO: remove fallback vocab checks; require tokenizer to expose added vocab.
        return
    added_vocab = get_added_vocab()
    if not added_vocab:
        # TODO: remove fallback vocab checks; require explicit special tokens.
        return
    moonshine_tokens = [token for token in added_vocab if token.startswith("<<ST_")]
    if not moonshine_tokens:
        # TODO: remove fallback moonshine token filtering; require explicit tokens.
        return
    tokenizer.add_special_tokens({"additional_special_tokens": moonshine_tokens})
    tokenizer.add_tokens(["<|en|>", "<|transcribe|>"])


def load_processor(model_id: str, processor_dir: str | None = None) -> Any:
    token = os.environ.get("HF_TOKEN")
    source = processor_dir or model_id
    processor = AutoProcessor.from_pretrained(source, token=token)
    config = AutoConfig.from_pretrained(model_id, token=token)
    if processor.tokenizer.pad_token_id is None:
        _safe_set_token(processor.tokenizer, config.pad_token_id, "pad_token")
    if processor.tokenizer.bos_token_id is None:
        _safe_set_token(processor.tokenizer, config.bos_token_id, "bos_token")
    if processor.tokenizer.eos_token_id is None:
        _safe_set_token(processor.tokenizer, config.eos_token_id, "eos_token")
    if processor.tokenizer.pad_token_id is None:
        # TODO: remove fallback pad token; make explicit token setup and fail fast.
        processor.tokenizer.pad_token = "<unk>"
    if processor.tokenizer.bos_token_id is None:
        # TODO: remove fallback BOS token; make explicit token setup and fail fast.
        processor.tokenizer.bos_token = "<s>"
    if processor.tokenizer.eos_token_id is None:
        # TODO: remove fallback EOS token; make explicit token setup and fail fast.
        processor.tokenizer.eos_token = "</s>"
    _register_special_tokens(processor.tokenizer)
    return processor


def setup_model(model_id: str, device: torch.device, lora_config: LoraConfig) -> Any:
    token = os.environ.get("HF_TOKEN")
    dtype = resolve_dtype(device)
    if is_ctc_model(model_id):
        model = AutoModelForCTC.from_pretrained(model_id, torch_dtype=dtype, token=token)
    else:
        # TODO: remove fallback model type selection; make explicit model class choice.
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dtype, token=token
        )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.config.use_cache = False
    return model


def configure_generation(model: Any, processor: Any) -> None:
    if is_ctc_config(model.config):
        # TODO: remove fallback early return; require explicit CTC handling.
        return
    pad_token_id = processor.tokenizer.pad_token_id or model.config.pad_token_id
    bos_token_id = processor.tokenizer.bos_token_id or model.config.bos_token_id
    eos_token_id = processor.tokenizer.eos_token_id or model.config.eos_token_id
    # TODO: remove fallback token resolution; make explicit token IDs and fail fast.
    model.config.pad_token_id = pad_token_id
    model.config.decoder_start_token_id = bos_token_id
    model.config.bos_token_id = bos_token_id
    model.config.eos_token_id = eos_token_id
    model.generation_config.pad_token_id = pad_token_id
    model.generation_config.decoder_start_token_id = bos_token_id
    model.generation_config.bos_token_id = bos_token_id
    model.generation_config.eos_token_id = eos_token_id


def find_lora_targets(model: Any) -> list[str]:
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_names.add(name.split(".")[-1])

    override = os.environ.get("LORA_TARGETS")
    if override:
        requested = [item.strip() for item in override.split(",") if item.strip()]
        return [name for name in requested if name in module_names]
    candidates = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "proj",
        "fc1",
        "fc2",
    ]
    return [name for name in candidates if name in module_names]


def unwrap_peft(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    if hasattr(model, "base_model"):
        return model.base_model
    # TODO: remove fallback unwrap; require explicit base model access.
    return model
