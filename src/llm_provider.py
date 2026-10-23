"""OpenAI configuration and credentials shared by the labeling stages."""

import os
from pathlib import Path

from openai import OpenAI


LLM_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"


def model_or_default(requested_model: str | None) -> str:
    """Use the requested OpenAI model or the project default."""
    return requested_model or DEFAULT_OPENAI_MODEL


def create_llm_client(*, api_key: str | None):
    """Create the OpenAI client used for Responses API structured outputs."""
    if not api_key:
        raise ValueError("Für OpenAI fehlt der API-Key.")
    return OpenAI(api_key=api_key)


def load_api_key(env_path: Path) -> str:
    """Read OPENAI_API_KEY from the environment or the project's .env file."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key

    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "OPENAI_API_KEY":
                api_key = value.strip().strip('"\'')
                if api_key:
                    return api_key

    raise RuntimeError(
        "OPENAI_API_KEY fehlt. Bitte den Schlüssel in .env eintragen."
    )
