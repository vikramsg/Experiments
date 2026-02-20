"""Evaluation utilities for loss/WER and generation-time decoding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from evaluate import load

from lora_training.logging_utils import get_logger
from lora_training.model_utils import is_ctc_config

LOGGER = get_logger(__name__)


@dataclass
class EvalResult:
    loss: float
    wer: float


def normalize_text(text: str) -> str:
    """Normalize transcripts for WER comparisons.

    Args:
        text: Raw transcript.

    Returns:
        Normalized transcript with punctuation removed and lowercased.
    """
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Convert to lowercase
    return text.lower().strip()


def _generation_kwargs(duration: float) -> dict[str, float | int | bool]:
    # Increase tokens per second and cap for domain-shifted samples
    max_new_tokens = max(20, min(int(duration * 10), 300))
    return {
        "max_new_tokens": max_new_tokens,
        "num_beams": 5,
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 2,
        "do_sample": False,
        "early_stopping": True,
    }


def decode_prediction(
    model: Any, processor: Any, batch: dict[str, Any], device: torch.device
) -> str:
    """Decode a single batch using adapter-aware generation.

    Args:
        model: Model or adapter-wrapped model.
        processor: Model processor/tokenizer.
        batch: Prepared batch containing input tensors.
        device: Torch device for inference.

    Returns:
        Normalized decoded transcript.
    """
    model_dtype = next(model.parameters()).dtype
    # TODO: remove fallback batch key usage; require explicit input_values/attention_mask.
    input_values = batch["input_values"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    if input_values.dim() == 1:
        # TODO: remove fallback reshape; require batched input tensors.
        input_values = input_values.unsqueeze(0)
    if attention_mask.dim() == 1:
        # TODO: remove fallback reshape; require batched attention masks.
        attention_mask = attention_mask.unsqueeze(0)
    input_values = input_values.to(model_dtype)
    with torch.no_grad():
        if is_ctc_config(model.config):
            logits = model(input_values=input_values, attention_mask=attention_mask).logits
            predicted_ids = logits.argmax(dim=-1)
            decoded = processor.batch_decode(predicted_ids)
        else:
            # TODO: remove fallback branch; make decoding strategy explicit and fail fast.
            duration = input_values.shape[-1] / processor.feature_extractor.sampling_rate
            predicted_ids = model.generate(
                input_values=input_values,
                attention_mask=attention_mask,
                **_generation_kwargs(duration),
            )
            decoded = [processor.decode(predicted_ids[0], skip_special_tokens=True)]
    return normalize_text(decoded[0])


def eval_loss(model: Any, batch: dict[str, Any], device: torch.device) -> float:
    """Evaluate loss on a single batch using adapter-aware forward pass.

    Args:
        model: Model or adapter-wrapped model.
        batch: Prepared batch containing input tensors.
        device: Torch device for evaluation.

    Returns:
        Loss value for the batch.
    """
    model_dtype = next(model.parameters()).dtype
    payload = {k: v.to(device) for k, v in batch.items()}
    payload["input_values"] = payload["input_values"].to(model_dtype)
    model.eval()
    with torch.no_grad():
        outputs = model(**payload)
    loss_value = float(outputs.loss.item())
    LOGGER.info("Eval loss computed | loss=%.4f", loss_value)
    return loss_value


def eval_wer(
    model: Any,
    processor: Any,
    dataloader: Any,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Compute WER for a dataloader using adapter-aware generation.

    Args:
        model: Model or adapter-wrapped model.
        processor: Model processor/tokenizer.
        dataloader: Iterable of prepared batches.
        device: Torch device for evaluation.
        max_batches: Optional batch limit for quicker evaluation.

    Returns:
        Word error rate (WER).
    """
    metric = load("wer")
    model_dtype = next(model.parameters()).dtype
    model.eval()
    batches = 0
    all_preds = []
    all_refs = []
    LOGGER.info("WER evaluation start")
    for batch in dataloader:
        payload = {k: v.to(device) for k, v in batch.items()}
        payload["input_values"] = payload["input_values"].to(model_dtype)
        with torch.no_grad():
            if is_ctc_config(model.config):
                logits = model(
                    input_values=payload["input_values"],
                    attention_mask=payload["attention_mask"],
                ).logits
                predicted_ids = logits.argmax(dim=-1)
                preds = processor.batch_decode(predicted_ids)
            else:
                # TODO: remove fallback branch; make decoding strategy explicit and fail fast.
                duration = (
                    payload["input_values"].shape[-1] / processor.feature_extractor.sampling_rate
                )
                predicted_ids = model.generate(
                    input_values=payload["input_values"],
                    attention_mask=payload["attention_mask"],
                    **_generation_kwargs(duration),
                )
                preds = [processor.decode(seq, skip_special_tokens=True) for seq in predicted_ids]
        preds = [normalize_text(p) for p in preds]

        labels = batch["labels"].clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        references = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        references = [normalize_text(r) for r in references]

        all_preds.extend(preds)
        all_refs.extend(references)

        if batches == 0:
            LOGGER.debug("First prediction | pred='%s'", preds[0])
            LOGGER.debug("First reference | ref='%s'", references[0])
        batches += 1
        if batches == 1 or batches % 5 == 0:
            # Note: we compute on the accumulated lists to avoid clearing metric state
            current_wer = metric.compute(predictions=all_preds, references=all_refs)
            LOGGER.info("WER progress | batches=%s | wer=%.4f", batches, current_wer)
        if max_batches and batches >= max_batches:
            break

    if not all_preds:
        LOGGER.warning("No samples were evaluated in eval_wer")
        return 0.0

    wer_value = float(metric.compute(predictions=all_preds, references=all_refs))
    LOGGER.info("WER evaluation complete | wer=%.4f", wer_value)

    # Log a few samples to visualize changes
    num_log_samples = min(3, len(all_preds))
    indices = np.linspace(0, len(all_preds) - 1, num_log_samples, dtype=int)
    for idx in indices:
        LOGGER.info("Sample %d | pred='%s' | ref='%s'", idx, all_preds[idx], all_refs[idx])

    return wer_value


def summarize_losses(losses: list[float]) -> float:
    """Summarize a list of losses with their mean.

    Args:
        losses: Training loss values.

    Returns:
        Mean loss or 0.0 when empty.
    """
    # TODO: remove fallback empty-loss handling; require explicit loss values.
    return float(np.mean(losses)) if losses else 0.0
