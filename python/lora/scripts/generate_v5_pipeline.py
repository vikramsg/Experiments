import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path so we can import lora_training logging
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lora_training.logging_utils import get_logger, setup_logging

# Initialize logging
log_path = Path("outputs/my_voice_tune_v5/pipeline.log")
setup_logging(log_path=log_path)
logger = get_logger("v5_pipeline")

JARGON = [
    "@refactor.md",
    "justfile",
    "uv run",
    "WER",
    "_posts",
    "@app",
    "/tmp",
    "nohup.out",
    "data/mixed_train.jsonl",
    "f5-tts-mlx",
    "lora_adapter",
    "moonshine-tiny",
]

TEMPLATES = [
    "Please check the {} file.",
    "Did you run {} yet?",
    "The {} is showing a high error rate.",
    "Look at the {} for more details.",
    "Update the {} with the new parameters.",
    "Why is {} failing?",
    "We need to decrease the {} significantly.",
    "Add {} to the configuration.",
    "The output is saved in {}.",
    "Ensure {} is properly formatted.",
]


def generate_prompts(num=500):
    prompts = set()
    while len(prompts) < num:
        jargon1 = random.choice(JARGON)
        jargon2 = random.choice(JARGON)
        if random.random() < 0.3 and jargon1 != jargon2:
            sentence = random.choice(TEMPLATES).format(f"{jargon1} and {jargon2}")
        else:
            sentence = random.choice(TEMPLATES).format(jargon1)
        prompts.add(sentence)
    return list(prompts)


def spell_out_for_tts(text: str) -> str:
    replacements = {
        "@": " at ",
        "_": " underscore ",
        "/": " slash ",
        ".md": " dot m d ",
        ".jsonl": " dot json l ",
        ".out": " dot out ",
        "uv": " u v ",
        "WER": " w e r ",
        "f5-tts-mlx": " f 5 t t s m l x ",
        "tmp": " temp ",
        "nohup": " no hup ",
        "justfile": " just file ",
    }
    spoken_text = text
    for symbol, spoken in replacements.items():
        spoken_text = spoken_text.replace(symbol, spoken)
    return " ".join(spoken_text.split())


def run_synthetic_generation(num_samples: int, manifest_path: str, audio_prefix: str):
    logger.info("Starting synthetic data generation for %d samples...", num_samples)
    import librosa
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from f5_tts_mlx.cfm import F5TTS
    from f5_tts_mlx.utils import convert_char_to_pinyin

    ref_audio_path = "data/raw_audio/my_voice/clip_ec16024b.wav"
    ref_audio_text = (
        "put a to do docstring at the top of the build domain manifest script about "
        "this issue. put in known issues document. Don't put a single or two liner, "
        "be a little more detailed."
    )

    logger.info("Loading F5TTS model...")
    f5tts = F5TTS.from_pretrained("lucasnewman/f5-tts-mlx")

    # CRITICAL FIX: The audio in the dataset is 16kHz! F5-TTS natively expects 24kHz.
    # If we feed 16kHz audio blindly, the model hears the conditioning voice
    # as 1.5x higher pitched and sped up (the "chipmunk/cartoon" effect).
    # We MUST resample to 24000 Hz.
    audio, _ = librosa.load(ref_audio_path, sr=24000)
    audio = mx.array(audio)
    target_rms = 0.1
    rms = mx.sqrt(mx.mean(mx.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms

    out_dir = Path("data/synthetic_audio")
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = generate_prompts(num_samples)
    manifest_entries = []

    HOP_LENGTH = 256

    logger.info("Generating %d synthetic audio files...", len(prompts))
    for i, target_text in enumerate(prompts):
        spoken_text = spell_out_for_tts(target_text)
        gen_text = convert_char_to_pinyin([ref_audio_text + " " + spoken_text])

        ref_audio_len = audio.shape[0] // HOP_LENGTH
        ref_text_len = len(ref_audio_text.encode("utf-8"))
        gen_text_len = len(spoken_text.encode("utf-8"))
        duration_in_frames = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)

        wave, _ = f5tts.sample(
            mx.expand_dims(audio, axis=0),
            text=gen_text,
            duration=duration_in_frames,
            steps=16,  # Enough for clarity but fast enough
            method="rk4",
            speed=1.0,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
        )

        wave = wave[audio.shape[0] :]
        mx.eval(wave)

        # NOTE: Save as 24kHz. Moonshine will resample it down to 16kHz during "
        # training internally anyway.
        out_file = out_dir / f"{audio_prefix}_{i:04d}.wav"
        sf.write(str(out_file), np.array(wave), 24000)

        manifest_entries.append({"audio": str(out_file), "text": target_text})

        if (i + 1) % 50 == 0 or (i + 1) == num_samples:
            logger.info("Generated %d/%d files...", i + 1, len(prompts))

    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")
    logger.info("Synthetic data generation for %s complete.", manifest_path)


def mix_datasets(synthetic_manifest: str, output_manifest: str):
    logger.info("Mixing datasets into %s...", output_manifest)
    mixed = []

    with open("data/my_voice_train.jsonl") as f:
        for line in f:
            mixed.append(json.loads(line))

    with open(synthetic_manifest) as f:
        for line in f:
            mixed.append(json.loads(line))

    random.shuffle(mixed)

    with open(output_manifest, "w") as f:
        for entry in mixed:
            f.write(json.dumps(entry) + "\n")
    logger.info("Dataset mixing complete. Total samples: %d", len(mixed))


def run_experiment_and_monitor(
    experiment_name: str, train_manifest: str, max_steps: int, eval_interval: int
):
    logger.info("Launching experiment %s...", experiment_name)

    args_str = (
        f"--model-id UsefulSensors/moonshine-tiny --train-manifest {train_manifest} "
        f"--eval-manifest data/my_voice_eval.jsonl --heldout-manifest data/heldout_manifest.jsonl "
        f"--max-steps {max_steps} --eval-interval {eval_interval} --learning-rate 1e-4 "
        "--lora-r 32 --lora-alpha 64 --lora-dropout 0.1 --use-dora"
    )
    cmd = ["just", "run-experiment", experiment_name, f'"{args_str}"']

    logger.info("Executing: %s", " ".join(cmd))
    subprocess.run(" ".join(cmd), shell=True, check=True)

    logger.info("Monitoring training logs for %s...", experiment_name)
    log_file = f"outputs/{experiment_name}/experiment.log"

    while not os.path.exists(log_file):
        time.sleep(2)

    with open(log_file) as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(5)
                status = subprocess.run(
                    f"just status {experiment_name}", shell=True, capture_output=True, text=True
                )
                if "No running process found" in status.stdout:
                    for rem_line in f.readlines():
                        logger.info(rem_line.strip())
                    break
                continue
            logger.info(line.strip())

    logger.info("Training %s complete.", experiment_name)


def run_verification(experiment_name: str):
    logger.info("Running transcription verification for %s...", experiment_name)
    transcribe_cmd = (
        f'just transcribe "--model-id UsefulSensors/moonshine-tiny '
        f"--adapter-dir outputs/{experiment_name}/lora_adapter_best "
        f"--manifest data/my_voice_eval.jsonl "
        f'--output outputs/{experiment_name}/eval_results.json"'
    )
    logger.info("Executing transcription: %s", transcribe_cmd)
    result = subprocess.run(transcribe_cmd, shell=True, capture_output=True, text=True)
    logger.info(result.stdout)
    if result.stderr:
        logger.error(result.stderr)
    return result.returncode == 0


def ensure_heldout_manifest():
    if not os.path.exists("data/heldout_manifest.jsonl"):
        logger.info("Creating dummy heldout_manifest.jsonl...")
        with open("data/heldout_manifest.jsonl", "w") as f:
            f.write(
                json.dumps(
                    {
                        "audio": "data/raw_audio/my_voice/clip_d0221d6e.wav",
                        "text": "What is the build manifest for?",
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    ensure_heldout_manifest()

    # --- PHASE 1: Proof of Concept ---
    logger.info("=== PHASE 1: PROOF OF CONCEPT ===")
    mini_synth_manifest = "data/synthetic_train_mini.jsonl"
    mini_mixed_manifest = "data/mixed_train_mini.jsonl"
    mini_exp_name = "my_voice_tune_v5_mini"

    run_synthetic_generation(
        num_samples=5, manifest_path=mini_synth_manifest, audio_prefix="synth_mini"
    )
    mix_datasets(synthetic_manifest=mini_synth_manifest, output_manifest=mini_mixed_manifest)
    run_experiment_and_monitor(
        experiment_name=mini_exp_name,
        train_manifest=mini_mixed_manifest,
        max_steps=50,
        eval_interval=25,
    )

    success = run_verification(experiment_name=mini_exp_name)
    if not success:
        logger.error("Phase 1 Verification Failed. Halting pipeline.")
        sys.exit(1)

    logger.info("Phase 1 verified successfully. Transitioning to Phase 2.")

    # --- PHASE 2: Full Scale ---
    logger.info("=== PHASE 2: FULL SCALE ===")
    full_synth_manifest = "data/synthetic_train.jsonl"
    full_mixed_manifest = "data/mixed_train.jsonl"
    full_exp_name = "my_voice_tune_v5"

    run_synthetic_generation(
        num_samples=505, manifest_path=full_synth_manifest, audio_prefix="synth_full"
    )
    mix_datasets(synthetic_manifest=full_synth_manifest, output_manifest=full_mixed_manifest)
    run_experiment_and_monitor(
        experiment_name=full_exp_name,
        train_manifest=full_mixed_manifest,
        max_steps=1000,
        eval_interval=100,
    )

    success = run_verification(experiment_name=full_exp_name)
    if not success:
        logger.error("Phase 2 Verification completed with errors.")
    else:
        logger.info("\n--- Pipeline Execution Completed Successfully ---")
