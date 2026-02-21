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

import torch
from evaluate import load
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    MoonshineForConditionalGeneration,
)

from lora_data.data_loader import (
    build_manifest_dataset,
    load_manifest,
    prepare_dataset,
)
from lora_training.evaluation import normalize_text
from lora_training.logging_utils import get_logger, setup_logging
from lora_training.model_utils import (
    choose_device,
    configure_generation,
    is_ctc_config,
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
    parser.add_argument("--processor-dir", default=None, help="Path to processor directory (optional)")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", help="Single audio file to transcribe")
    input_group.add_argument("--manifest", help="JSONL manifest with audio array/paths")
    
    parser.add_argument("--output", help="Output JSON report path")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default=None)
    return parser.parse_args()


def transcribe(args: argparse.Namespace) -> SttReport:
    setup_logging()
    device = choose_device(args.device)
    LOGGER.info("Device selected | device=%s", device)
    
    # Load processor
    processor_path = args.processor_dir if args.processor_dir else args.model_id
    processor = load_processor(args.model_id, processor_path)
    LOGGER.info("Processor loaded | path=%s", processor_path)
    
    # Load base model
    config = AutoConfig.from_pretrained(args.model_id)
    if is_ctc_config(config):
        base_model = AutoModelForCTC.from_pretrained(args.model_id)
    elif config.model_type == "moonshine":
        base_model = MoonshineForConditionalGeneration.from_pretrained(args.model_id)
    elif config.model_type in {"whisper", "speech-encoder-decoder"}:
        base_model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id)
    else:
        raise ValueError(f"Unsupported model_type: {config.model_type}")
        
    # Apply adapter if provided
    if args.adapter_dir:
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
        LOGGER.info("LoRA adapter applied | adapter_dir=%s", args.adapter_dir)
    else:
        model = base_model
        LOGGER.info("No adapter provided. Running baseline inference.")
        
    model.to(device)
    model.eval()
    configure_generation(model, processor)

    # Load data
    entries = []
    if args.audio:
        entries = [{"audio": args.audio}]
        LOGGER.info("Transcribing single audio file | path=%s", args.audio)
    elif args.manifest:
        manifest_path = Path(args.manifest)
        entries = load_manifest(manifest_path)
        LOGGER.info("Loaded manifest | path=%s | samples=%s", manifest_path, len(entries))

    # To use existing prepare_dataset, we create a dataset
    # We might need to ensure text is present (even if empty) for prepare_dataset not to crash if it expects it
    for entry in entries:
        if "text" not in entry:
            entry["text"] = ""
            
    dataset = build_manifest_dataset(entries)
    prepared = prepare_dataset(dataset, processor)
    
    metric = load("wer")
    samples: list[SttSample] = []
    has_references = any(bool(e.get("text")) for e in entries)
    
    for idx, item in enumerate(prepared):
        if idx == 0 or (idx + 1) % 10 == 0:
            LOGGER.info("Inference progress | sample=%s/%s", idx + 1, len(prepared))
            
        entry = entries[idx]
        reference_text = entry.get("text", "")
        
        if config.model_type == "moonshine":
            audio = normalize_audio_rms(entry["audio"])
            inputs = processor(
                audio,
                sampling_rate=processor.feature_extractor.sampling_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            # Safe attention mask passing depending on the model config
            duration = len(audio) / processor.feature_extractor.sampling_rate
            max_new_tokens = max(10, min(int(duration * 5), 150))
            
            kwargs = {
                "input_values": input_values,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "num_beams": 5,
                "repetition_penalty": 1.3,
                "no_repeat_ngram_size": 2,
                "do_sample": False,
                "early_stopping": True,
            }
            
            with torch.no_grad():
                predicted_ids = model.generate(**kwargs)
            prediction = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
        else:
            if "input_features" in item:
                input_key = "input_features"
            elif "input_values" in item:
                input_key = "input_values"
            else:
                raise KeyError("Item does not contain 'input_features' or 'input_values'")
                
            input_values = torch.tensor(item[input_key]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(item["attention_mask"]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if is_ctc_config(config):
                    raise NotImplementedError("CTC decoding is explicitly unsupported.")
                predicted_ids = model.generate(
                    **{
                        input_key: input_values,
                        "attention_mask": attention_mask,
                    },
                )
                prediction = processor.decode(predicted_ids[0], skip_special_tokens=True)
                
        prediction_norm = normalize_text(prediction)
        reference_norm = normalize_text(reference_text) if reference_text else None
        
        sample = SttSample(
            index=idx,
            prediction=prediction,
            reference=reference_text if reference_text else None,
            audio_path=entry.get("audio") if isinstance(entry.get("audio"), str) else None
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
        # If no output specified, just dump to stdout cleanly
        print(json.dumps(asdict(report), indent=2))
        
    return report

def main() -> None:
    args = parse_args()
    transcribe(args)

if __name__ == "__main__":
    main()
