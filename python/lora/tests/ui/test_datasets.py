import uuid

import pytest

from db.models import Dataset, DatasetRecord, Record
from ui.core import db_client


@pytest.fixture
def mock_dataset():
    unique_id = uuid.uuid4().hex[:8]
    unique_name = f"mock_train_{unique_id}"
    unique_file = f"mock_audio_{unique_id}.wav"

    with db_client.session_scope() as session:
        ds = Dataset(name=unique_name, description="A mock dataset for testing")
        session.add(ds)
        session.flush()

        rec = Record(
            data_type="AUDIO",
            file_path=unique_file,
            content="this is a test record",
        )
        session.add(rec)
        session.flush()

        ds_rec = DatasetRecord(dataset_id=ds.id, record_id=rec.id)
        session.add(ds_rec)
        session.flush()


def test_datasets_page_renders_with_data(test_client, mock_dataset):
    response = test_client.get("/datasets")
    assert response.status_code == 200
    assert "mock_train_" in response.text
    assert "A mock dataset for testing" in response.text
    assert "this is a test record" in response.text
    assert "mock_audio_" not in response.text  # Usually not showing raw path


def test_audio_endpoint_returns_404_for_missing_file(test_client, mock_dataset):
    with db_client.session_scope() as session:
        rec = session.query(Record).first()
        record_id = rec.id

    response = test_client.get(f"/audio/{record_id}")
    assert response.status_code == 404
    assert "File missing on disk" in response.text
