import os
from threading import Lock

from dotenv import load_dotenv

from src.constants.dataset import FINNHUB_API_KEY_ENV
from src.constants.filtering import LLM_PREMIUM_KEY_ENV, LLM_STANDARD_KEY_ENV
from src.utils.app_config import AppConfig


class ConfigProvider:
    """Loads and caches the secrets read from the environment."""

    _config: AppConfig | None = None
    _lock = Lock()

    @classmethod
    def get(cls) -> AppConfig:
        """Return the process-wide configuration, loading it once on first use.

        Only secrets are read here (from the environment); every non-secret
        setting is a constant under src/constants/ and is imported directly
        where it is used. Absent secrets are left as None and validated by the
        component that needs them.

        Returns:
            AppConfig: The loaded application configuration.
        """
        if cls._config is None:
            with cls._lock:
                if cls._config is None:
                    load_dotenv()
                    cls._config = AppConfig(
                        finnhub_api_key=os.getenv(FINNHUB_API_KEY_ENV),
                        ai_api_key=os.getenv(LLM_STANDARD_KEY_ENV),
                        premium_ai_api_key=os.getenv(LLM_PREMIUM_KEY_ENV),
                    )
        return cls._config
