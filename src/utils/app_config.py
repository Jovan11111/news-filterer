from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    # Dataset
    dataset_output_dir: Path

    # Finnhub news fetcher
    finnhub_api_key: str
    finnhub_rate_limit_delay: float
    finnhub_timeout: float
    finnhub_max_retries: int
    finnhub_chunk_days: int
    finnhub_page_cap: int
    finnhub_min_interval: float
