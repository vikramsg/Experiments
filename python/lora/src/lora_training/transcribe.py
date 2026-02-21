"""Standalone STT transcription tool.

Supports transcribing a single audio file or a batch of files via a JSONL manifest.
Can run baseline inference or use a trained LoRA adapter.

Usage:
    uv run python src/lora_training/transcribe.py \
        --model-id UsefulSensors/moonshine-tiny \
        --manifest data/heldout_manifest.jsonl \
        --output outputs/artifact_test.json

    uv run python src/lora_training/transcribe.py \
        --model-id UsefulSensors/moonshine-tiny \
        --audio my_audio.wav
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from evaluate import load
from peft import PeftModel
from transformers import (
    AutoConfig,
    MoonshineForConditionalGeneration,
    PreTrainedModel,
)

from lora_data.data_loader import (
    build_manifest_dataset,
    load_manifest,
    normalize_audio,
    prepare_dataset,
)
from lora_training.evaluation import normalize_text
from lora_training.logging_utils import get_logger, setup_logging
from lora_training.model_utils import (
    choose_device,
    configure_generation,
    load_processor,
    normalize_audio_rms,
)

LOGGER = get_logger(__name__)


@dataclass
class SttSample:
    index: int
    prediction: str
    reference: str | None = None
    audio_path: str | None = None


@dataclass
class SttReport:
    model_id: str
    adapter_dir: str | None
    processor_dir: str | None
    device: str
    wer: float | None
    samples: list[SttSample]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STT transcription.")
    parser.add_argument("--model-id", required=True, help="Base model ID")
    parser.add_argument("--adapter-dir", default=None, help="Path to LoRA adapter directory")
    parser.add_argument(
        "--processor-dir", default=None, help="Path to processor directory (optional)"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", help="Single audio file to transcribe")
    input_group.add_argument("--manifest", help="JSONL manifest with audio array/paths")

    parser.add_argument("--output", help="Output JSON report path")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default=None)
    return parser.parse_args()


def load_inference_model(
    model_id: str, adapter_dir: str | None, device: torch.device
) -> tuple[PreTrainedModel, Any]:
    """Load the base model and optionally apply a LoRA adapter.

    Args:
        model_id: The base model identifier.
        adapter_dir: Path to the LoRA adapter directory, or None for baseline.
        device: The target execution device.

    Returns:
        A tuple of (model, config).
    """
    config = AutoConfig.from_pretrained(model_id)
    base_model = MoonshineForConditionalGeneration.from_pretrained(model_id)

    if adapter_dir:
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        LOGGER.info("LoRA adapter applied | adapter_dir=%s", adapter_dir)
    else:
        model = base_model
        LOGGER.info("No adapter provided. Running baseline inference.")

    model.to(device)
    model.eval()
    return model, config


def run_moonshine_inference(
    model: PreTrainedModel, processor: Any, audio: list[float], device: torch.device
) -> str:
    """Run inference using the Moonshine-specific processing path.

    Args:
        model: The trained/loaded model.
        processor: The model processor.
        audio: Raw audio waveform.
        device: Target execution device.

    Returns:
        The decoded transcription string.
    """
    audio_norm = normalize_audio_rms(audio)
    inputs = processor(
        audio_norm,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    duration = len(audio_norm) / processor.feature_extractor.sampling_rate
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
    return processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def transcribe(args: argparse.Namespace) -> SttReport:
    setup_logging()
    device = choose_device(args.device)
    LOGGER.info("Device selected | device=%s", device)

    processor_path = args.processor_dir if args.processor_dir else args.model_id
    processor = load_processor(args.model_id, processor_path)
    LOGGER.info("Processor loaded | path=%s", processor_path)

    model, config = load_inference_model(args.model_id, args.adapter_dir, device)
    configure_generation(model, processor)

    entries = []
    if args.audio:
        entries = [{"audio": args.audio}]
        LOGGER.info("Transcribing single audio file | path=%s", args.audio)
    elif args.manifest:
        manifest_path = Path(args.manifest)
        entries = load_manifest(manifest_path)
        LOGGER.info("Loaded manifest | path=%s | samples=%s", manifest_path, len(entries))

    # Ensure "text" exists so HF datasets prepare_dataset doesn't complain about missing keys
    for entry in entries:
        if "text" not in entry:
            entry["text"] = ""

    dataset = build_manifest_dataset(entries)
    prepared = prepare_dataset(dataset, processor)

    metric = load("wer")
    samples: list[SttSample] = []
    has_references = any(bool(e.get("text")) for e in entries)

    for idx, _ in enumerate(prepared):
        if idx == 0 or (idx + 1) % 10 == 0:
            LOGGER.info("Inference progress | sample=%s/%s", idx + 1, len(prepared))

        entry = entries[idx]
        reference_text = entry.get("text", "")

        audio_array = normalize_audio(entry["audio"])
        prediction = run_moonshine_inference(model, processor, audio_array, device)

        prediction_norm = normalize_text(prediction)
        reference_norm = normalize_text(reference_text) if reference_text else None

        sample = SttSample(
            index=idx,
            prediction=prediction,
            reference=reference_text if reference_text else None,
            audio_path=entry.get("audio") if isinstance(entry.get("audio"), str) else None,
        )
        samples.append(sample)

        if reference_norm:
            metric.add_batch(
                predictions=[prediction_norm],
                references=[reference_norm],
            )
            LOGGER.debug("Sample %d | pred='%s' | ref='%s'", idx, prediction_norm, reference_norm)
        else:
            LOGGER.info("Sample %d | pred='%s'", idx, prediction)

    wer_value = float(metric.compute()) if has_references else None

    report = SttReport(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        processor_dir=args.processor_dir,
        device=str(device),
        wer=wer_value,
        samples=samples,
    )

    if args.output:
        Path(args.output).write_text(json.dumps(asdict(report), indent=2))
        LOGGER.info("Report saved | output=%s", args.output)
        if wer_value is not None:
            LOGGER.info("Computed WER | wer=%.4f", wer_value)
    else:
        print(json.dumps(asdict(report), indent=2))

    return report


def main() -> None:
    args = parse_args()
    transcribe(args)


if __name__ == "__main__":
    main()
