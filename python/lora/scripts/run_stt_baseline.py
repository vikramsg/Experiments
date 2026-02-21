import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from lora_data.data_loader import build_manifest_dataset, load_manifest, prepare_dataset, normalize_audio
from lora_training.logging_utils import get_logger, setup_logging
from lora_training.model_utils import (
    choose_device,
    configure_generation,
    load_processor,
    normalize_audio_rms,
)
from transformers import MoonshineForConditionalGeneration

LOGGER = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--audio-list", required=True)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_args()
    device = choose_device(args.device)
    
    processor = load_processor(args.model_id, args.model_id)
    manifest_path = Path(args.audio_list)
    entries = load_manifest(manifest_path)
    
    model = MoonshineForConditionalGeneration.from_pretrained(args.model_id).to(device)
    model.eval()
    configure_generation(model, processor)

    results = []
    for idx, entry in enumerate(entries):
        audio_array = normalize_audio(entry["audio"])
        audio_rms = normalize_audio_rms(audio_array)
        inputs = processor(
            audio_rms,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            predicted_ids = model.generate(
                input_values=input_values,
                attention_mask=attention_mask,
                max_new_tokens=150,
            )
            
        prediction = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        results.append((entry["text"], prediction))
        
    print("\n\n--- RESULTS ---")
    for ref, pred in results:
        print(f"- **Expected:** `{ref}`")
        print(f"  - **Moonshine:** `{pred}`")

if __name__ == "__main__":
    main()
