from pathlib import Path

import pytest

from db.client import DBClient
from db.models import Dataset, Run
from lora_training.runners.experiment import (
    RegisteredRun,
    _finalize_experiment_run,
    _register_experiment_run,
    _resolve_dataset_id,
)
from lora_training.training_config import ExperimentConfig


def test_resolve_dataset_id(tmp_path: Path):
    db_file = tmp_path / "test.sqlite"
    client = DBClient(db_path=db_file)
    client.init_db()

    with client.session_scope() as session:
        ds = Dataset(name="test_dataset", description="")
        session.add(ds)
        session.commit()

        # Test valid DB dataset
        ds_id = _resolve_dataset_id(session, "db://test_dataset")
        assert ds_id == ds.id

        # Test local path
        assert _resolve_dataset_id(session, "data.jsonl") is None

        # Test missing DB dataset
        assert _resolve_dataset_id(session, "db://nonexistent") is None

        # Test invalid schema
        with pytest.raises(ValueError):
            _resolve_dataset_id(session, "s3://bucket/data")


def test_register_and_finalize_run(tmp_path: Path, monkeypatch):
    import db.client as client_module

    db_file = tmp_path / "test.sqlite"
    client = DBClient(db_path=db_file)
    client.init_db()

    # Monkeypatch the DBClient module directly
    monkeypatch.setattr(client_module, "DBClient", lambda: client)

    config = ExperimentConfig(
        model_id="test",
        output_dir=str(tmp_path / "outputs"),
        max_steps=10,
        eval_interval=5,
        dataset_path="data.jsonl",
        manifest_path="eval.jsonl",
        learning_rate=1e-4,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        batch_size=1,
        gradient_accumulation_steps=8,
        seed=42,
        safety_manifest_path=None,
        max_seconds=20.0,
        device=None,
        wer_stop_threshold=None,
        use_dora=False,
        init_lora_weights="gaussian",
        lora_module_filter=None,
        lora_targets=None,
    )

    reg_run = _register_experiment_run(config)
    assert isinstance(reg_run, RegisteredRun)
    assert reg_run.run_id > 0
    assert reg_run.eval_ds_id is None

    with client.session_scope() as session:
        run = session.get(Run, reg_run.run_id)
        assert run.status == "RUNNING"
        assert run.name == "outputs"

    _finalize_experiment_run(reg_run.run_id, config)
    with client.session_scope() as session:
        run = session.get(Run, reg_run.run_id)
        assert run.status == "COMPLETED"
        assert run.artifacts_dir == str(tmp_path / "outputs" / "lora_adapter")

    # Test failure case
    error = ValueError("Boom")
    _finalize_experiment_run(reg_run.run_id, config, error)
    with client.session_scope() as session:
        run = session.get(Run, reg_run.run_id)
        assert run.status == "FAILED"
        assert "Boom" in run.error_traceback
