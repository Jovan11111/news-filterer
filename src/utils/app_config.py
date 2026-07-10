from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    # Secrets read from the environment (absent ones are None and are validated
    # by whichever component actually needs them).
    finnhub_api_key: str | None
    ai_api_key: str | None
    premium_ai_api_key: str | None

    # Optional LLM endpoint overrides (e.g. an Azure AI Foundry deployment).
    # Absent ones fall back to the OpenAI defaults in src/constants/filtering.py.
    ai_base_url: str | None = None
    premium_ai_base_url: str | None = None
    ai_model: str | None = None
    premium_ai_model: str | None = None
    ai_api_version: str | None = None
    ai_use_responses: bool = False
    premium_ai_use_responses: bool = False
