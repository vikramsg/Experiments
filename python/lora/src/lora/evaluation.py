from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from evaluate import load

from lora.model_utils import unwrap_peft


@dataclass
class EvalResult:
    loss: float
    wer: float


def normalize_text(text: str) -> str:
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Convert to lowercase
    return text.lower().strip()


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
    return normalize_text(
        processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    )


def eval_loss(model: Any, batch: dict[str, Any], device: torch.device) -> float:
    base_model = unwrap_peft(model)
    model_dtype = next(model.parameters()).dtype
    payload = {k: v.to(device) for k, v in batch.items()}
    payload["input_values"] = payload["input_values"].to(model_dtype)
    base_model.eval()
    with torch.no_grad():
        outputs = base_model(**payload)
    return float(outputs.loss.item())


def eval_wer(
    model: Any,
    processor: Any,
    dataloader: Any,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    metric = load("wer")
    base_model = unwrap_peft(model)
    model_dtype = next(model.parameters()).dtype
    base_model.eval()
    batches = 0
    for batch in dataloader:
        payload = {k: v.to(device) for k, v in batch.items()}
        payload["input_values"] = payload["input_values"].to(model_dtype)
        with torch.no_grad():
            predicted_ids = base_model.generate(
                input_values=payload["input_values"],
                attention_mask=payload["attention_mask"],
            )
        preds = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        preds = [normalize_text(p) for p in preds]

        labels = batch["labels"].clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        references = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        references = [normalize_text(r) for r in references]

        metric.add_batch(predictions=preds, references=references)
        if batches == 0:
            print(f"DEBUG: Pred: '{preds[0]}'")
            print(f"DEBUG: Ref: '{references[0]}'")
        batches += 1
        if max_batches and batches >= max_batches:
            break
    return float(metric.compute())


def summarize_losses(losses: list[float]) -> float:
    return float(np.mean(losses)) if losses else 0.0
