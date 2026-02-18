from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from evaluate import load

from lora.logging_utils import get_logger
from lora.model_utils import is_ctc_config, unwrap_peft

LOGGER = get_logger(__name__)


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
        if is_ctc_config(base_model.config):
            logits = base_model(
                input_values=input_values, attention_mask=attention_mask
            ).logits
            predicted_ids = logits.argmax(dim=-1)
            decoded = processor.batch_decode(predicted_ids)
        else:
            # TODO: remove fallback branch; make decoding strategy explicit and fail fast.
            duration = input_values.shape[-1] / processor.feature_extractor.sampling_rate
            max_new_tokens = max(10, min(int(duration * 5), 150))
            predicted_ids = base_model.generate(
                input_values=input_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=5,
                repetition_penalty=1.3,
                no_repeat_ngram_size=2,
                do_sample=False,
                early_stopping=True,
            )
            decoded = [processor.decode(predicted_ids[0], skip_special_tokens=True)]
    return normalize_text(decoded[0])


def eval_loss(model: Any, batch: dict[str, Any], device: torch.device) -> float:
    base_model = unwrap_peft(model)
    model_dtype = next(model.parameters()).dtype
    payload = {k: v.to(device) for k, v in batch.items()}
    payload["input_values"] = payload["input_values"].to(model_dtype)
    base_model.eval()
    with torch.no_grad():
        outputs = base_model(**payload)
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
    metric = load("wer")
    base_model = unwrap_peft(model)
    model_dtype = next(model.parameters()).dtype
    base_model.eval()
    batches = 0
    LOGGER.info("WER evaluation start")
    for batch in dataloader:
        payload = {k: v.to(device) for k, v in batch.items()}
        payload["input_values"] = payload["input_values"].to(model_dtype)
        with torch.no_grad():
            if is_ctc_config(base_model.config):
                logits = base_model(
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
                max_new_tokens = max(10, min(int(duration * 5), 150))
                predicted_ids = base_model.generate(
                    input_values=payload["input_values"],
                    attention_mask=payload["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    num_beams=5,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=2,
                    do_sample=False,
                    early_stopping=True,
                )
                preds = [processor.decode(seq, skip_special_tokens=True) for seq in predicted_ids]
        preds = [normalize_text(p) for p in preds]

        labels = batch["labels"].clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        references = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        references = [normalize_text(r) for r in references]

        metric.add_batch(predictions=preds, references=references)
        if batches == 0:
            LOGGER.debug("First prediction | pred='%s'", preds[0])
            LOGGER.debug("First reference | ref='%s'", references[0])
        batches += 1
        if batches == 1 or batches % 5 == 0:
            LOGGER.info("WER progress | batches=%s", batches)
        if max_batches and batches >= max_batches:
            break
    wer_value = float(metric.compute())
    LOGGER.info("WER evaluation complete | wer=%.4f", wer_value)
    return wer_value


def summarize_losses(losses: list[float]) -> float:
    # TODO: remove fallback empty-loss handling; require explicit loss values.
    return float(np.mean(losses)) if losses else 0.0
