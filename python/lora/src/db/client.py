"""Database client using SQLAlchemy for MLOps tracking."""

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base

DEFAULT_DB_PATH = Path("data/tracker.sqlite")

_local = threading.local()

class DBClient:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{self.db_path}"
        else:
            url = "sqlite:///:memory:"
        
        # SQLite needs check_same_thread=False for cross-thread access, 
        # and poolclass=NullPool can be useful, but session handles it nicely
        self.engine = create_engine(url, connect_args={"check_same_thread": False})
        self.SessionFactory = sessionmaker(bind=self.engine)

    def init_db(self) -> None:
        """Create all tables."""
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
