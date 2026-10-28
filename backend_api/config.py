"""Configuration loaded from environment variables.

The API deliberately has no machine-specific addresses or credentials in source
code.  See ``backend_api/README.md`` for the supported variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env(primary: str, *aliases: str, default: str | None = None) -> str | None:
    """Read the documented setting first, retaining harmless legacy aliases."""
    for name in (primary, *aliases):
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _positive_int(value: str | None, default: int, name: str) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be at least 1")
    return parsed


def _resolve_path(value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the FastAPI service."""

    app_name: str
    environment: str
    log_level: str
    database_path: Path
    model_path: Path
    model_metadata_path: Path
    max_log_records: int
    cors_origins: Tuple[str, ...]
    enable_docs: bool

    @classmethod
    def from_environment(cls) -> "Settings":
        origins = tuple(
            origin.strip()
            for origin in (_env("CTD_CORS_ORIGINS", "THREAT_CORS_ORIGINS", default="http://localhost:8501") or "").split(",")
            if origin.strip()
        )
        return cls(
            app_name=_env("CTD_APP_NAME", default="Real-Time Cyber Threat Detection API") or "Real-Time Cyber Threat Detection API",
            environment=_env("CTD_ENV", default="development") or "development",
            log_level=(_env("CTD_LOG_LEVEL", "THREAT_LOG_LEVEL", default="INFO") or "INFO").upper(),
            database_path=_resolve_path(
                _env("CTD_DATABASE_PATH", "THREAT_DATABASE_PATH"),
                PROJECT_ROOT / "backend_api" / "data" / "traffic.db",
            ),
            model_path=_resolve_path(
                _env("CTD_MODEL_PATH", "THREAT_MODEL_PATH"),
                PROJECT_ROOT / "ai_model" / "artifacts" / "intrusion_detection_model.joblib",
            ),
            model_metadata_path=_resolve_path(
                _env("CTD_MODEL_METADATA_PATH", "THREAT_MODEL_METADATA_PATH"),
                PROJECT_ROOT / "ai_model" / "artifacts" / "model_metadata.json",
            ),
            max_log_records=_positive_int(
                _env("CTD_MAX_LOG_RECORDS", "THREAT_MAX_LOG_RECORDS"),
                1000,
                "CTD_MAX_LOG_RECORDS",
            ),
            cors_origins=origins,
            enable_docs=_as_bool(_env("CTD_ENABLE_DOCS"), default=True),
        )
