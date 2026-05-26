"""Configuration loading utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_env(key: str, default: str | None = None, required: bool = False) -> str | None:
    """Read an environment variable with optional required-flag."""
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"Required env var {key} is not set")
    return value


class Settings:
    """Centralized application settings."""

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data"
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    CONFIGS_DIR = PROJECT_ROOT / "configs"

    OPENAI_API_KEY = get_env("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = get_env("ANTHROPIC_API_KEY")
    XAI_API_KEY = get_env("XAI_API_KEY")
    XAI_BASE_URL = get_env("XAI_BASE_URL", "https://api.x.ai/v1")
    LLM_MODEL = get_env("LLM_MODEL", "grok-3-mini")
    LLM_PROVIDER = get_env("LLM_PROVIDER", "xai")
    MLFLOW_TRACKING_URI = get_env("MLFLOW_TRACKING_URI", "./mlruns")
    LOG_LEVEL = get_env("LOG_LEVEL", "INFO")


settings = Settings()
