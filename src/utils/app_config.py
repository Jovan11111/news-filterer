from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    # Secrets read from the environment (absent ones are None and are validated
    # by whichever component actually needs them).
    finnhub_api_key: str | None
    ai_api_key: str | None
    premium_ai_api_key: str | None
