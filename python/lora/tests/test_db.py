"""Tests for the SQLAlchemy database models and logic."""

from pathlib import Path
from db.client import DBClient
from db.models import Dataset, Record, Run, RunParam, RunMetric, DatasetRecord

def test_db_init_and_datasets(tmp_path: Path):
    db_file = tmp_path / "test.sqlite"
    client = DBClient(db_path=db_file)
    client.init_db()

    with client.session_scope() as session:
        # Create a run
        run = Run(name="gen_run", run_type="GENERATION", status="COMPLETED")
        session.add(run)
        session.flush()

        # Add records
        r1 = Record(source_run_id=run.id, data_type="AUDIO", file_path="audio1.wav", content="test 1")
        r2 = Record(source_run_id=run.id, data_type="AUDIO", file_path="audio2.wav", content="test 2")
        session.add_all([r1, r2])
        session.flush()

        assert r1.id is not None

        # Create dataset
        ds = Dataset(name="test_ds", description="Test dataset")
        session.add(ds)
        session.flush()

        dr1 = DatasetRecord(dataset_id=ds.id, record_id=r1.id)
        dr2 = DatasetRecord(dataset_id=ds.id, record_id=r2.id)
        session.add_all([dr1, dr2])

    with client.session_scope() as session:
        samples = session.query(Record).join(DatasetRecord).join(Dataset).filter(Dataset.name == "test_ds").all()
        assert len(samples) == 2
        assert samples[0].file_path == "audio1.wav"

def test_experiment_tracking(tmp_path: Path):
    db_file = tmp_path / "test2.sqlite"
    client = DBClient(db_path=db_file)
    client.init_db()

    with client.session_scope() as session:
        run = Run(name="train_run", run_type="TRAINING", status="RUNNING", log_file_path="log.txt")
        session.add(run)
        session.flush()

        param = RunParam(run_id=run.id, key="lr", value="0.001")
        session.add(param)
        
        metric = RunMetric(run_id=run.id, step=100, key="wer", value=0.1)
        session.add(metric)
        session.flush()

        assert metric.id is not None

    with client.session_scope() as session:
        r = session.query(Run).filter(Run.name == "train_run").first()
        assert r is not None
        assert r.status == "RUNNING"
        assert len(r.params) == 1
        assert len(r.metrics) == 1
        assert r.metrics[0].value == 0.1

        r.status = "COMPLETED"

    with client.session_scope() as session:
        r2 = session.query(Run).filter(Run.name == "train_run").first()
        assert r2.status == "COMPLETED"
