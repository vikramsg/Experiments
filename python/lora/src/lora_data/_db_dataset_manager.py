"""Database Dataset Manager for SQLite data provenance."""

import json
import os
from pathlib import Path

from db.client import DBClient
from db.models import Dataset, DatasetRecord, Record
from lora_training.logging_utils import get_logger

LOGGER = get_logger(__name__)

def hash_file(filepath: str | Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(str(filepath), "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def ensure_heldout_manifest_db() -> int:
    """Create a dummy safety heldout dataset if missing and load to DB."""
    client = DBClient()
    client.init_db()

    heldout_path = Path("data/heldout_manifest.jsonl")
    if not heldout_path.exists():
        LOGGER.info("Creating dummy heldout_manifest.jsonl...")
        heldout_path.parent.mkdir(parents=True, exist_ok=True)
        with open(heldout_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "audio": "data/raw_audio/my_voice/clip_d0221d6e.wav",
                        "text": "What is the build manifest for?",
                    }
                )
                + "\n"
            )

    with client.session_scope() as session:
        ds = session.query(Dataset).filter_by(name="heldout_eval").first()
        if ds:
            session.query(DatasetRecord).filter_by(dataset_id=ds.id).delete()
        else:
            ds = Dataset(name="heldout_eval", description="Heldout dataset for safety eval")
            session.add(ds)
            session.flush()

        with open(heldout_path) as f:
            for line in f:
                entry = json.loads(line)
                filepath = entry["audio"]
                text = entry["text"]
                file_hash = hash_file(filepath) if os.path.exists(filepath) else ""

                rec = session.query(Record).filter_by(file_path=filepath).first()
                if not rec:
                    rec = Record(
                        file_path=filepath,
                        content=text,
                        data_type="AUDIO",
                        metadata_json={"source_type": "REAL"},
                        file_hash=file_hash,
                    )
                    session.add(rec)
                    session.flush()

                session.add(DatasetRecord(dataset_id=ds.id, record_id=rec.id))
        ds_id = ds.id
    return ds_id

def load_eval_to_db() -> int:
    """Migrate the real voice eval manifest into DB."""
    client = DBClient()
    client.init_db()

    with client.session_scope() as session:
        ds = session.query(Dataset).filter_by(name="real_voice_eval").first()
        if ds:
            session.query(DatasetRecord).filter_by(dataset_id=ds.id).delete()
        else:
            ds = Dataset(name="real_voice_eval", description="Real voice evaluation dataset")
            session.add(ds)
            session.flush()

        with open("data/my_voice_eval.jsonl") as f:
            for line in f:
                entry = json.loads(line)
                filepath = entry["audio"]
                text = entry["text"]
                file_hash = hash_file(filepath) if os.path.exists(filepath) else ""

                rec = session.query(Record).filter_by(file_path=filepath).first()
                if not rec:
                    rec = Record(
                        file_path=filepath,
                        content=text,
                        data_type="AUDIO",
                        metadata_json={"source_type": "REAL"},
                        file_hash=file_hash,
                    )
                    session.add(rec)
                    session.flush()

                session.add(DatasetRecord(dataset_id=ds.id, record_id=rec.id))
        ds_id = ds.id
    return ds_id

def create_real_voice_dataset() -> int:
    """Migrate the real voice train manifest into DB."""
    client = DBClient()
    client.init_db()

    with client.session_scope() as session:
        ds = session.query(Dataset).filter_by(name="real_voice_train").first()
        if ds:
            session.query(DatasetRecord).filter_by(dataset_id=ds.id).delete()
        else:
            ds = Dataset(name="real_voice_train", description="Real voice training dataset")
            session.add(ds)
            session.flush()

        with open("data/my_voice_train.jsonl") as f:
            for line in f:
                entry = json.loads(line)
                filepath = entry["audio"]
                text = entry["text"]
                file_hash = hash_file(filepath) if os.path.exists(filepath) else ""

                rec = session.query(Record).filter_by(file_path=filepath).first()
                if not rec:
                    rec = Record(
                        file_path=filepath,
                        content=text,
                        data_type="AUDIO",
                        metadata_json={"source_type": "REAL"},
                        file_hash=file_hash,
                    )
                    session.add(rec)
                    session.flush()

                session.add(DatasetRecord(dataset_id=ds.id, record_id=rec.id))
        ds_id = ds.id
    return ds_id

def mix_db_datasets(dataset1_id: int, dataset2_id: int, mixed_name: str) -> int:
    """Combine two DB datasets into a new logical dataset."""
    client = DBClient()
    client.init_db()

    with client.session_scope() as session:
        records = (
            session.query(DatasetRecord)
            .filter(DatasetRecord.dataset_id.in_([dataset1_id, dataset2_id]))
            .all()
        )
        record_ids = {r.record_id for r in records}

        ds = session.query(Dataset).filter_by(name=mixed_name).first()
        if ds:
            session.query(DatasetRecord).filter_by(dataset_id=ds.id).delete()
        else:
            ds = Dataset(
                name=mixed_name, description=f"Mixed dataset from {dataset1_id} and {dataset2_id}"
            )
            session.add(ds)
            session.flush()

        for rid in record_ids:
            session.add(DatasetRecord(dataset_id=ds.id, record_id=rid))

        ds_id = ds.id
    return ds_id
