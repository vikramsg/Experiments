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
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from evaluate import load
from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq

from datasets import Dataset

from lora.data_loader import load_manifest, normalize_audio, prepare_dataset
from lora.model_utils import choose_device, configure_generation, load_processor


@dataclass
class SttSample:
    index: int
    reference: str
    prediction: str


@dataclass
class SttReport:
    model_id: str
    adapter_dir: str
    processor_dir: str
    device: str
    wer: float
    samples: list[SttSample]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STT using saved LoRA artifacts")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--processor-dir", required=True)
    parser.add_argument("--audio-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default=None)
    return parser.parse_args()


def run_stt(args: argparse.Namespace) -> SttReport:
    device = choose_device(args.device)
    processor = load_processor(args.model_id, args.processor_dir)
    manifest_path = Path(args.audio_list)
    entries = load_manifest(manifest_path)
    if not entries:
        raise ValueError("Audio manifest is empty")
    records = {
        "audio": [normalize_audio(item["audio"]) for item in entries],
        "text": [item["text"] for item in entries],
        "speaker_id": [item.get("speaker_id", -1) for item in entries],
    }
    dataset = Dataset.from_dict(records)

    prepared = prepare_dataset(dataset, processor)
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.to(device)
    configure_generation(model, processor)
    metric = load("wer")
    samples: list[SttSample] = []

    for idx, item in enumerate(prepared):
        input_values = torch.tensor(item["input_values"]).unsqueeze(0)
        attention_mask = torch.tensor(item["attention_mask"]).unsqueeze(0)
        batch = {
            "input_values": input_values,
            "attention_mask": attention_mask,
        }
        with torch.no_grad():
            predicted_ids = model.generate(
                input_values=batch["input_values"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
        prediction = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        reference = processor.tokenizer.decode(
            item["labels"][item["labels"] != -100], skip_special_tokens=True
        )
        metric.add_batch(predictions=[prediction], references=[reference])
        samples.append(SttSample(index=idx, reference=reference, prediction=prediction))

    report = SttReport(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        processor_dir=args.processor_dir,
        device=str(device),
        wer=float(metric.compute()),
        samples=samples,
    )
    Path(args.output).write_text(json.dumps(asdict(report), indent=2))
    return report


def main() -> None:
    args = parse_args()
    run_stt(args)


if __name__ == "__main__":
    main()
