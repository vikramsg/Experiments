from typing import Optional
from pydantic import validator
from pydantic_settings import BaseSettings
from pydantic.networks import PostgresDsn


class DBConfig(BaseSettings):
    postgres_user: str
    postgres_password: str
    postgres_db: str

    postgres_url: Optional[PostgresDsn] = None

    @validator("postgres_url")
    def create_db_url(cls, _, values) -> Optional[PostgresDsn]:
        if "postgres_user" in values:
            url = f"""postgresql://{values["postgres_user"]}:{values["postgres_password"]}@db/{values["postgres_db"]}"""
            return PostgresDsn(url=url)
        else:
            return None
