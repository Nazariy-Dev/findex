"""Конфіг лише з оточення (12-factor, правило III)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    index_path: Path = Path("index.bin")
    log_level: str = "INFO"
    search_mode: str = "def"  # "def" або "async" — лише для заміру в M2
