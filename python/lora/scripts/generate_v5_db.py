import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from db.client import DBClient
from lora_training.logging_utils import get_logger, setup_logging

# Initialize logging
log_path = Path("outputs/my_voice_tune_db/pipeline.log")
setup_logging(log_path=log_path)
logger = get_logger("v5_pipeline_db")

JARGON = [
    "@refactor.md", "justfile", "uv run", "WER", "_posts", 
    "@app", "/tmp", "nohup.out", "data/mixed_train.jsonl",
    "f5-tts-mlx", "lora_adapter", "moonshine-tiny"
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
    "Ensure {} is properly formatted."
]

def hash_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def generate_prompts(num=500, seed=42):
    random.seed(seed)
    prompts = set()
    while len(prompts) < num:
        jargon1 = random.choice(JARGON)
        jargon2 = random.choice(JARGON)
        if random.random() < 0.3 and jargon1 != jargon2:
            sentence = random.choice(TEMPLATES).format(f"{jargon1} and {jargon2}")
        else:
            sentence = random.choice(TEMPLATES).format(jargon1)
        prompts.add(sentence)
    
    # Sort to ensure determinism after set conversion
    return sorted(list(prompts))

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
        "justfile": " just file "
    }
    spoken_text = text
    for symbol, spoken in replacements.items():
        spoken_text = spoken_text.replace(symbol, spoken)
    return " ".join(spoken_text.split())

def run_synthetic_generation_db(num_samples: int, dataset_name: str, audio_prefix: str, seed: int = 42):
    logger.info("Starting DB-tracked synthetic generation for %d samples...", num_samples)
    
    client = DBClient()
    client.init_db()
    
    # Log the generation run
    run_config = {
        "num_samples": num_samples,
        "seed": seed,
        "audio_prefix": audio_prefix,
        "jargon": JARGON,
        "templates": TEMPLATES
    }
    gen_run = client.start_generation_run(f"synth_{audio_prefix}", run_config, str(log_path))
    run_id = gen_run.id
    
    try:
        import librosa
        import mlx.core as mx
        import numpy as np
        import soundfile as sf
        from f5_tts_mlx.cfm import F5TTS
        from f5_tts_mlx.utils import convert_char_to_pinyin

        ref_audio_path = "data/raw_audio/my_voice/clip_ec16024b.wav"
        ref_audio_text = "put a to do docstring at the top of the build domain manifest script about this issue. put in known issues document. Don't put a single or two liner, be a little more detailed."
        
        logger.info("Loading F5TTS model...")
        f5tts = F5TTS.from_pretrained("lucasnewman/f5-tts-mlx")
        
        audio, sr = librosa.load(ref_audio_path, sr=24000)
        audio = mx.array(audio)
        target_rms = 0.1
        rms = mx.sqrt(mx.mean(mx.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
            
        out_dir = Path("data/synthetic_audio")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        prompts = generate_prompts(num_samples, seed)
        HOP_LENGTH = 256
        
        sample_ids = []
        
        logger.info("Generating %d synthetic audio files...", len(prompts))
        for i, target_text in enumerate(prompts):
            spoken_text = spell_out_for_tts(target_text)
            gen_text = convert_char_to_pinyin([ref_audio_text + " " + spoken_text])
            
            ref_audio_len = audio.shape[0] // HOP_LENGTH
            ref_text_len = len(ref_audio_text.encode('utf-8'))
            gen_text_len = len(spoken_text.encode('utf-8'))
            duration_in_frames = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)
            
            out_file = out_dir / f"{audio_prefix}_{i:04d}.wav"
            
            if not out_file.exists():
                wave, _ = f5tts.sample(
                    mx.expand_dims(audio, axis=0),
                    text=gen_text,
                    duration=duration_in_frames,
                    steps=16,
                    method="rk4",
                    speed=1.0,
                    cfg_strength=2.0,
                    sway_sampling_coef=-1.0
                )
                
                wave = wave[audio.shape[0] :]
                mx.eval(wave)
                
                sf.write(str(out_file), np.array(wave), 24000)
            
            file_hash = hash_file(str(out_file))
            # Just an approximation of duration
            duration = float(librosa.get_duration(path=str(out_file))) if out_file.exists() else 0.0
            
            sample = client.log_audio_sample(
                file_path=str(out_file),
                target_text=target_text,
                source_type="SYNTHETIC",
                duration=duration,
                file_hash=file_hash,
                run_id=run_id
            )
            sample_ids.append(sample.id)
            
            if (i + 1) % 50 == 0 or (i + 1) == num_samples:
                logger.info("Generated %d/%d files...", i + 1, len(prompts))
                
        # Create dataset
        ds = client.create_dataset(dataset_name, f"Synthetic dataset {audio_prefix}", sample_ids)
        client.finish_generation_run(run_id, "COMPLETED")
        logger.info("Synthetic data generation complete. Dataset ID: %d", ds.id)
        return ds.id
        
    except Exception as e:
        logger.error("Generation failed", exc_info=True)
        if run_id:
            client.finish_generation_run(run_id, "FAILED")
        raise e

def create_real_voice_dataset():
    """Migrate the real voice train manifest into DB."""
    client = DBClient()
    client.init_db()
    sample_ids = []
    
    with open("data/my_voice_train.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            filepath = entry["audio"]
            text = entry["text"]
            file_hash = hash_file(filepath) if os.path.exists(filepath) else ""
            
            sample = client.log_audio_sample(
                file_path=filepath,
                target_text=text,
                source_type="REAL",
                duration=0.0,
                file_hash=file_hash,
                run_id=None
            )
            sample_ids.append(sample.id)
            
    ds = client.create_dataset("real_voice_train", "Real voice training dataset", sample_ids)
    return ds.id

def mix_db_datasets(dataset1_id: int, dataset2_id: int, mixed_name: str) -> int:
    client = DBClient()
    client.init_db()
    conn = client._get_conn()
    cur = conn.cursor()
    
    # Get all samples from both datasets
    cur.execute("SELECT audio_sample_id FROM dataset_samples WHERE dataset_id = ? OR dataset_id = ?", (dataset1_id, dataset2_id))
    samples = set(row["audio_sample_id"] for row in cur.fetchall())
    
    ds = client.create_dataset(mixed_name, f"Mixed dataset from {dataset1_id} and {dataset2_id}", list(samples))
    return ds.id

def load_eval_to_db():
    client = DBClient()
    client.init_db()
    sample_ids = []
    
    with open("data/my_voice_eval.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            filepath = entry["audio"]
            text = entry["text"]
            file_hash = hash_file(filepath) if os.path.exists(filepath) else ""
            
            sample = client.log_audio_sample(
                file_path=filepath,
                target_text=text,
                source_type="REAL",
                duration=0.0,
                file_hash=file_hash,
                run_id=None
            )
            sample_ids.append(sample.id)
            
    ds = client.create_dataset("real_voice_eval", "Real voice evaluation dataset", sample_ids)
    return ds.id

def ensure_heldout_manifest_db():
    client = DBClient()
    client.init_db()
    if not os.path.exists("data/heldout_manifest.jsonl"):
        logger.info("Creating dummy heldout_manifest.jsonl...")
        with open("data/heldout_manifest.jsonl", "w") as f:
            f.write(json.dumps({"audio": "data/raw_audio/my_voice/clip_d0221d6e.wav", "text": "What is the build manifest for?"}) + "\n")
            
    sample_ids = []
    with open("data/heldout_manifest.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            filepath = entry["audio"]
            text = entry["text"]
            file_hash = hash_file(filepath) if os.path.exists(filepath) else ""
            sample = client.log_audio_sample(
                file_path=filepath,
                target_text=text,
                source_type="REAL",
                duration=0.0,
                file_hash=file_hash,
                run_id=None
            )
            sample_ids.append(sample.id)
    
    ds = client.create_dataset("heldout_eval", "Heldout dataset for safety eval", sample_ids)
    return ds.id

def run_experiment_and_monitor(experiment_name: str, train_dataset: str, eval_dataset: str, heldout_dataset: str, max_steps: int, eval_interval: int):
    logger.info("Launching experiment %s...", experiment_name)
    
    cmd = [
        "just", "run-experiment", experiment_name,
        f'"--model-id UsefulSensors/moonshine-tiny --dataset-path db://{train_dataset} --manifest-path db://{eval_dataset} --safety-manifest-path db://{heldout_dataset} --max-steps {max_steps} --eval-interval {eval_interval} --learning-rate 1e-4 --lora-r 32 --lora-alpha 64 --lora-dropout 0.1 --use-dora"'
    ]
    
    logger.info("Executing: %s", ' '.join(cmd))
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
                status = subprocess.run(f"just status {experiment_name}", shell=True, capture_output=True, text=True)
                if "No running process found" in status.stdout:
                    for rem_line in f.readlines():
                        logger.info(rem_line.strip())
                    break
                continue
            logger.info(line.strip())
            
    logger.info("Training %s complete.", experiment_name)

if __name__ == "__main__":
    logger.info("=== PHASE 1: DB MINI PROOF OF CONCEPT ===")
    
    heldout_id = ensure_heldout_manifest_db()
    eval_id = load_eval_to_db()
    real_id = create_real_voice_dataset()
    
    # Generate 5 mini samples
    synth_mini_id = run_synthetic_generation_db(num_samples=2, dataset_name="synth_mini_db", audio_prefix="synth_mini_db")
    
    mixed_mini_id = mix_db_datasets(real_id, synth_mini_id, "mixed_mini_db")
    
    # Run Experiment
    run_experiment_and_monitor(
        experiment_name="my_voice_tune_v5_mini_db", 
        train_dataset="mixed_mini_db", 
        eval_dataset="real_voice_eval", 
        heldout_dataset="heldout_eval",
        max_steps=5, 
        eval_interval=2
    )

