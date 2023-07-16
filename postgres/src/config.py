from functools import lru_cache
from pydantic import Field
from pydantic import BaseSettings
from pydantic.networks import PostgresDsn
import os


class Settings(BaseSettings):
    postgres_user: str = Field(..., env="POSTGRES_USER")
    postgres_password: str = Field(..., env="POSTGRES_PASSWORD")
    postgres_server: str = Field(..., env="POSTGRES_HOST")
    postgres_db: str = Field(..., env="POSTGRES_DB")

    @property
    def postgres_dsn(self) -> PostgresDsn:
        return PostgresDsn.build(
            scheme="postgresql",
            user=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_server,
            path=f"/{self.postgres_db}",
        )


@lru_cache()
def get_config() -> Settings:
    return Settings(_env_file=os.getenv("ENVFILE", ".env"), _env_file_encoding="utf-8")  # type: ignore
