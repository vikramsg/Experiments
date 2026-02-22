"""Migrate legacy JSONL raw audio manifests into the SQLite tracker."""

import argparse
import json
import logging
from pathlib import Path

import librosa

from db.client import DBClient
from db.models import Dataset, DatasetRecord, Record

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def hash_file(filepath: str | Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(str(filepath), "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def migrate_manifest(manifest_path: str, dataset_name: str, description: str) -> None:
    path = Path(manifest_path)
    if not path.exists():
        logger.warning(f"Manifest not found, skipping: {path}")
        return

    client = DBClient()
    client.init_db()

    with client.session_scope() as session:
        ds = session.query(Dataset).filter_by(name=dataset_name).first()
        if ds:
            logger.info(f"Dataset '{dataset_name}' already exists. Clearing old records.")
            session.query(DatasetRecord).filter_by(dataset_id=ds.id).delete()
        else:
            ds = Dataset(name=dataset_name, description=description)
            session.add(ds)
            session.flush()

        logger.info(f"Migrating {path} -> DB Dataset '{dataset_name}'...")
        count = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                filepath = entry["audio"]
                text = entry["text"]

                if not Path(filepath).exists():
                    logger.warning(f"Audio file missing, skipping record: {filepath}")
                    continue

                file_hash = hash_file(filepath)
                
                # Check if this exact file is already in Records
                rec = session.query(Record).filter_by(file_path=filepath).first()
                if not rec:
                    # Calculate duration for real data
                    duration = float(librosa.get_duration(path=filepath))
                    
                    rec = Record(
                        file_path=filepath,
                        content=text,
                        data_type="AUDIO",
                        metadata_json={"source_type": "REAL", "duration_sec": duration},
                        file_hash=file_hash,
                    )
                    session.add(rec)
                    session.flush()

                # Link it to the dataset
                session.add(DatasetRecord(dataset_id=ds.id, record_id=rec.id))
                count += 1

        logger.info(f"Successfully migrated {count} records into '{dataset_name}'.")

def main():
    parser = argparse.ArgumentParser(description="Migrate JSONL manifests to DB.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to JSONL file")
    parser.add_argument("--dataset-name", type=str, required=True, help="Target DB Dataset Name")
    parser.add_argument("--description", type=str, default="Migrated from JSONL", help="Description")
    args = parser.parse_args()
    
    migrate_manifest(args.manifest, args.dataset_name, args.description)

if __name__ == "__main__":
    main()
