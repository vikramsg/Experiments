"""Test the db dataset manager."""

from db.client import DBClient
from db.models import Dataset, Record
from lora_data._db_dataset_manager import ensure_heldout_manifest_db


def test_ensure_heldout_manifest_db(tmp_path, monkeypatch):
    import lora_data._db_dataset_manager as manager
    
    db_file = tmp_path / "test.sqlite"
    client = DBClient(db_path=db_file)
    client.init_db()

    # Monkeypatch so ensure_heldout_manifest_db uses our temporary test client
    monkeypatch.setattr(manager, "DBClient", lambda: client)

    ds_id = ensure_heldout_manifest_db()
    assert ds_id > 0

    with client.session_scope() as session:
        ds = session.get(Dataset, ds_id)
        assert ds.name == "heldout_eval"

        records = session.query(Record).all()
        assert len(records) > 0
        assert records[0].data_type == "AUDIO"
