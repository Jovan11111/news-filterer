import os
from threading import Lock

from dotenv import load_dotenv

from src.constants.dataset import (
    DATASET_OUTPUT_DIR,
    FINNHUB_API_KEY_ENV,
    FINNHUB_CHUNK_DAYS,
    FINNHUB_MAX_RETRIES,
    FINNHUB_PAGE_CAP,
    FINNHUB_RATE_LIMIT_DELAY_SECONDS,
    FINNHUB_REQUESTS_PER_MINUTE,
    FINNHUB_TIMEOUT_SECONDS,
)
from src.utils.app_config import AppConfig


class ConfigProvider:
    """Loads and caches the process-wide application configuration."""

    _config: AppConfig | None = None
    _lock = Lock()

    @classmethod
    def get(cls) -> AppConfig:
        """Return the process-wide configuration, loading it once on first use.

        The only value read from the environment is the secret Finnhub API key;
        every other setting is a constant defined in src/constants/dataset.py.

        Returns:
            AppConfig: The loaded application configuration.
        """
        if cls._config is None:
            with cls._lock:
                if cls._config is None:
                    load_dotenv()
                    cls._config = AppConfig(
                        dataset_output_dir=DATASET_OUTPUT_DIR,
                        finnhub_api_key=cls.__require_env(FINNHUB_API_KEY_ENV),
                        finnhub_rate_limit_delay=FINNHUB_RATE_LIMIT_DELAY_SECONDS,
                        finnhub_timeout=FINNHUB_TIMEOUT_SECONDS,
                        finnhub_max_retries=FINNHUB_MAX_RETRIES,
                        finnhub_chunk_days=FINNHUB_CHUNK_DAYS,
                        finnhub_page_cap=FINNHUB_PAGE_CAP,
                        finnhub_min_interval=60.0 / max(1, FINNHUB_REQUESTS_PER_MINUTE),
                    )
        return cls._config

    @staticmethod
    def __require_env(name: str) -> str:
        """Return the value of a required environment variable or raise if it is unset."""
        value = os.getenv(name)
        if not value:
            raise ValueError(f"Missing required environment variable: {name}")
        return value
