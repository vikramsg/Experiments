from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import get_config

config = get_config()
url = config.postgres_dsn

# Create a SQLAlchemy engine
engine = create_engine(url)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def is_task_exists():
    ...
