import os
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

from src.utils.app_config import AppConfig


class ConfigProvider:
    """Loads and caches the process-wide application configuration."""

    _config: AppConfig | None = None
    _lock = Lock()

    @classmethod
    def get(cls) -> AppConfig:
        """Return the process-wide configuration, loading it once on first use.

        Returns:
            AppConfig: The loaded application configuration.
        """
        if cls._config is None:
            with cls._lock:
                if cls._config is None:
                    load_dotenv()
                    cls._config = AppConfig(
                        dataset_output_dir=Path(
                            os.getenv("DATASET_OUTPUT_DIR", "data")
                        ),
                        finnhub_api_key=cls.__require_env("FINNHUB_API_KEY"),
                        finnhub_rate_limit_delay=float(
                            os.getenv("FINNHUB_RATE_LIMIT_DELAY", "1.0")
                        ),
                        finnhub_timeout=float(os.getenv("FINNHUB_TIMEOUT", "20.0")),
                        finnhub_max_retries=int(os.getenv("FINNHUB_MAX_RETRIES", "3")),
                        finnhub_chunk_days=int(os.getenv("FINNHUB_CHUNK_DAYS", "7")),
                        finnhub_page_cap=int(os.getenv("FINNHUB_PAGE_CAP", "250")),
                        finnhub_min_interval=60.0 / max(1, int(os.getenv("FINNHUB_REQUESTS_PER_MINUTE", "50"))),
                    )
        return cls._config

    @staticmethod
    def __require_env(name: str) -> str:
        """Return the value of a required environment variable or raise if it is unset."""
        value = os.getenv(name)
        if not value:
            raise ValueError(f"Missing required environment variable: {name}")
        return value
