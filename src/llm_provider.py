"""Small OpenAI structured-output client factory shared by the pipeline."""

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
