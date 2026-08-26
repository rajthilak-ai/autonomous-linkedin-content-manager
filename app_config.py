"""Application configuration validation shared by the CLI and Streamlit UI."""

from __future__ import annotations

import os


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing."""


def validate_environment() -> None:
    """Ensure all required API keys are available before running the crew."""
    required_keys = ("OPENAI_API_KEY", "SERPER_API_KEY")
    missing = [key for key in required_keys if not os.getenv(key)]

    if missing:
        raise ConfigurationError(
            "Missing required environment variable(s): "
            f"{', '.join(missing)}. Set them in a local .env file, or in "
            "Streamlit Cloud App settings → Secrets (see "
            ".streamlit/secrets.toml.example)."
        )
