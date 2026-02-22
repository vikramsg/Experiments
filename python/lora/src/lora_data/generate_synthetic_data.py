"""CLI orchestrator for generating synthetic datasets and tracking them in SQLite."""

import argparse
from pathlib import Path

from db.client import DBClient
from db.models import Dataset, DatasetRecord, Record, Run, RunParam
from lora_data._db_dataset_manager import (
    create_real_voice_dataset,
    ensure_heldout_manifest_db,
    hash_file,
    load_eval_to_db,
    mix_db_datasets,
)
from lora_data._jargon_prompt_generator import generate_prompts, spell_out_for_tts
from lora_data._tts_engine import F5TTSEngine
from lora_training.logging_utils import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic voice datasets.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of files to generate")
    parser.add_argument("--dataset-name", type=str, required=True, help="DB dataset output name")
    parser.add_argument("--audio-prefix", type=str, required=True, help="Prefix for .wav files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--mix-with-real", action="store_true", help="Combine with real train data")
    parser.add_argument("--mixed-name", type=str, default="", help="Name for the mixed dataset")
    return parser.parse_args()


def generate_synthetic_data(
    num_samples: int, dataset_name: str, audio_prefix: str, seed: int = 42
) -> int:
    """Generate audio files using F5TTS and log records into the local tracker."""
    log_path = Path(f"outputs/gen_{audio_prefix}/generation.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_path=log_path)
    logger = get_logger("synthetic_generator")

    logger.info("Starting DB-tracked synthetic generation for %d samples...", num_samples)

    client = DBClient()
    client.init_db()

    with client.session_scope() as session:
        run = Run(
            name=f"synth_{audio_prefix}",
            run_type="GENERATION",
            status="RUNNING",
            log_file_path=str(log_path),
        )
        session.add(run)
        session.flush()

        run.params.append(RunParam(key="num_samples", value=str(num_samples)))
        run.params.append(RunParam(key="seed", value=str(seed)))
        run.params.append(RunParam(key="audio_prefix", value=audio_prefix))

        run_id = run.id

    try:
        ref_audio_path = "data/raw_audio/my_voice/clip_ec16024b.wav"
        ref_audio_text = (
            "put a to do docstring at the top of the build domain manifest script about "
            "this issue. put in known issues document. Don't put a single or two liner, "
            "be a little more detailed."
        )

        engine = F5TTSEngine(ref_audio_path, ref_audio_text)

        out_dir = Path("data/synthetic_audio")
        out_dir.mkdir(parents=True, exist_ok=True)

        prompts = generate_prompts(num_samples, seed)
        sample_ids = []

        logger.info("Generating %d synthetic audio files...", len(prompts))
        for i, target_text in enumerate(prompts):
            spoken_text = spell_out_for_tts(target_text)
            out_file = out_dir / f"{audio_prefix}_{i:04d}.wav"

            duration = engine.synthesize_audio(spoken_text, out_file)
            file_hash = hash_file(out_file)

            with client.session_scope() as session:
                sample = Record(
                    file_path=str(out_file),
                    content=target_text,
                    data_type="AUDIO",
                    metadata_json={"duration_sec": duration, "source_type": "SYNTHETIC"},
                    file_hash=file_hash,
                    source_run_id=run_id,
                )
                session.add(sample)
                session.flush()
                sample_ids.append(sample.id)

            if (i + 1) % 50 == 0 or (i + 1) == num_samples:
                logger.info("Generated %d/%d files...", i + 1, len(prompts))

        with client.session_scope() as session:
            ds = session.query(Dataset).filter_by(name=dataset_name).first()
            if ds:
                session.query(DatasetRecord).filter_by(dataset_id=ds.id).delete()
            else:
                ds = Dataset(name=dataset_name, description=f"Synthetic dataset {audio_prefix}")
                session.add(ds)
            session.flush()

            ds_id = ds.id
            for sid in sample_ids:
                session.add(DatasetRecord(dataset_id=ds.id, record_id=sid))

            run_obj = session.query(Run).get(run_id)
            if run_obj:
                run_obj.status = "COMPLETED"

        logger.info("Synthetic data generation complete. Dataset ID: %d", ds_id)
        return ds_id

    except Exception as e:
        logger.error("Generation failed", exc_info=True)
        with client.session_scope() as session:
            run_obj = session.query(Run).get(run_id)
            if run_obj:
                run_obj.status = "FAILED"
                import traceback

                run_obj.error_traceback = traceback.format_exc()
        raise e


def main() -> None:
    args = parse_args()

    # Pre-setup standard real datasets so they are trackable by DB if missing
    ensure_heldout_manifest_db()
    load_eval_to_db()
    real_id = create_real_voice_dataset()

    synth_id = generate_synthetic_data(
        num_samples=args.num_samples,
        dataset_name=args.dataset_name,
        audio_prefix=args.audio_prefix,
        seed=args.seed,
    )

    if args.mix_with_real and args.mixed_name:
        mix_db_datasets(real_id, synth_id, args.mixed_name)
        print(f"Success! Mixed dataset '{args.mixed_name}' created.")
    else:
        print(f"Success! Synthetic dataset '{args.dataset_name}' created.")


if __name__ == "__main__":
    main()
