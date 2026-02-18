"""Run STT inference with a saved LoRA adapter and processor artifacts.

Usage:
    uv run python scripts/run_stt.py \
        --model-id UsefulSensors/moonshine-tiny \
        --adapter-dir outputs/real_small/lora_adapter \
        --processor-dir outputs/real_small/processor \
        --audio-list data/heldout_manifest.jsonl \
        --output outputs/real_small/artifact_test.json \
        --device mps

Flags:
    --model-id       Base model ID
    --adapter-dir    LoRA adapter directory
    --processor-dir  Processor directory
    --audio-list     JSONL manifest with audio arrays and text
    --output         Output JSON report path
    --device         Device override (mps, cuda, cpu)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import Dataset
from evaluate import load
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    MoonshineForConditionalGeneration,
)

from lora.data_loader import load_manifest, normalize_audio, prepare_dataset
from lora.logging_utils import get_logger, setup_logging
from lora.model_utils import (
    choose_device,
    configure_generation,
    is_ctc_config,
    load_processor,
    normalize_audio_rms,
)


@dataclass
class SttSample:
    index: int
    reference: str
    prediction: str


@dataclass
class SttReport:
    model_id: str
    adapter_dir: str | None
    processor_dir: str
    device: str
    wer: float
    samples: list[SttSample]


LOGGER = get_logger(__name__)


def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower().strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STT using saved LoRA artifacts")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--processor-dir", required=True)
    parser.add_argument("--audio-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default=None)
    return parser.parse_args()


def run_stt(args: argparse.Namespace) -> SttReport:
    setup_logging()
    device = choose_device(args.device)
    LOGGER.info("Device selected | device=%s", device)
    processor = load_processor(args.model_id, args.processor_dir)
    LOGGER.info("Processor loaded | model_id=%s", args.model_id)
    config = AutoConfig.from_pretrained(args.model_id)
    manifest_path = Path(args.audio_list)
    entries = load_manifest(manifest_path)
    if not entries:
        raise ValueError("Audio manifest is empty")
    LOGGER.info("Loaded manifest | path=%s | samples=%s", manifest_path, len(entries))
    records = {
        "audio": [normalize_audio(item["audio"]) for item in entries],
        "text": [item["text"] for item in entries],
        "speaker_id": [item.get("speaker_id", -1) for item in entries],
    }
    dataset = Dataset.from_dict(records)

    prepared = prepare_dataset(dataset, processor)
    LOGGER.info("Prepared dataset | samples=%s", len(prepared))
    if is_ctc_config(config):
        base_model = AutoModelForCTC.from_pretrained(args.model_id)
    elif config.model_type == "moonshine":
        base_model = MoonshineForConditionalGeneration.from_pretrained(args.model_id)
    else:
        base_model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id)
    if args.adapter_dir:
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
        adapter_dir = args.adapter_dir
    else:
        model = base_model
        adapter_dir = None
    LOGGER.info("Model loaded | adapter=%s", adapter_dir or "none")
    model.to(device)
    configure_generation(model, processor)
    metric = load("wer")
    samples: list[SttSample] = []

    for idx, item in enumerate(prepared):
        if idx == 0 or (idx + 1) % 10 == 0:
            LOGGER.info("Inference progress | sample=%s/%s", idx + 1, len(prepared))
        if config.model_type == "moonshine":
            entry = entries[idx]
            audio = normalize_audio_rms(entry["audio"])
            inputs = processor(
                audio,
                sampling_rate=processor.feature_extractor.sampling_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)
            duration = len(audio) / processor.feature_extractor.sampling_rate
            max_new_tokens = max(10, min(int(duration * 5), 150))
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=5,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=2,
                    do_sample=False,
                    early_stopping=True,
                )
            prediction = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            reference = entry["text"]
        else:
            input_key = "input_features" if "input_features" in item else "input_values"
            input_values = torch.tensor(item[input_key]).unsqueeze(0)
            attention_mask = torch.tensor(item["attention_mask"]).unsqueeze(0)
            batch = {
                input_key: input_values,
                "attention_mask": attention_mask,
            }
            with torch.no_grad():
                if is_ctc_config(config):
                    logits = model(
                        **{
                            input_key: batch[input_key].to(device),
                            "attention_mask": batch["attention_mask"].to(device),
                        }
                    ).logits
                    predicted_ids = logits.argmax(dim=-1)
                    prediction = processor.batch_decode(predicted_ids)[0]
                else:
                    predicted_ids = model.generate(
                        **{
                            input_key: batch[input_key].to(device),
                            "attention_mask": batch["attention_mask"].to(device),
                        },
                    )
                    prediction = processor.decode(predicted_ids[0], skip_special_tokens=True)
            label_ids = item["labels"]
            if hasattr(label_ids, "tolist"):
                label_ids = label_ids.tolist()
            label_ids = [token for token in label_ids if token != -100]
            reference = processor.tokenizer.decode(label_ids, skip_special_tokens=True)
        metric.add_batch(
            predictions=[normalize_text(prediction)],
            references=[normalize_text(reference)],
        )
        samples.append(SttSample(index=idx, reference=reference, prediction=prediction))

    report = SttReport(
        model_id=args.model_id,
        adapter_dir=adapter_dir,
        processor_dir=args.processor_dir,
        device=str(device),
        wer=float(metric.compute()),
        samples=samples,
    )
    Path(args.output).write_text(json.dumps(asdict(report), indent=2))
    LOGGER.info("Report saved | output=%s | wer=%.4f", args.output, report.wer)
    return report


def main() -> None:
    args = parse_args()
    run_stt(args)


if __name__ == "__main__":
    main()
