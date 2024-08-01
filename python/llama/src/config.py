import os
from functools import lru_cache

from pydantic import BaseSettings, Field


class LLAMAConfig(BaseSettings):
    llam_bin_file: str = Field(..., env="LLAMA_BIN_FILE")


@lru_cache()
def get_llama_config() -> LLAMAConfig:
    return LLAMAConfig(_env_file=os.getenv("ENVFILE", ".env"), _env_file_encoding="utf-8")  # type: ignore
