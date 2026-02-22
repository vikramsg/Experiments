import os

import pytest
from fasthtml.common import Client

# Override the DB_URL to use in-memory SQLite for testing
os.environ["DB_URL"] = "sqlite:///:memory:"

from ui.core import db_client
from ui.main import app


@pytest.fixture(autouse=True)
def setup_test_db():
    from db.models import Base

    db_client.init_db()
    yield
    with db_client.engine.begin() as conn:
        Base.metadata.drop_all(conn)


@pytest.fixture
def test_client():
    return Client(app)
