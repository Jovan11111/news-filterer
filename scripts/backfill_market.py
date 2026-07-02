import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date

from src.constants.dataset import MARKET_INDEX_SYMBOL
from src.dataset.price_fetcher import PriceFetcher
from src.utils.news_database import NewsDatabase
from src.utils.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.joinpath("data")


def _datasets_to_process() -> list[Path]:
    """Resolve which database files the backfill should touch (every .db in data/)."""
    found = sorted(DATA_DIR.glob("*.db"))
    if not found:
        raise SystemExit(f"No dataset databases found in {DATA_DIR}.")
    return found


def _market_changes(dates: list[date]) -> dict[date, tuple[float | None, float | None]]:
    """Compute the 1-/7-day S&P 500 change for each news date."""
    fetcher = PriceFetcher()
    fetcher.load_history(MARKET_INDEX_SYMBOL, min(dates), max(dates))
    changes: dict[date, tuple[float | None, float | None]] = {}
    for news_date in dates:
        price = fetcher.price(MARKET_INDEX_SYMBOL, news_date)
        changes[news_date] = (price.change_1_day, price.change_7_days)
    return changes


def _backfill(path: Path):
    """Add and populate the market-change columns for a single dataset."""
    db = NewsDatabase(path)
    try:
        dates = db.distinct_news_dates()
        if not dates:
            logger.warning(f"{path.name} has no news rows; skipping.")
            return
        logger.info(f"Backfilling {path.name}: {len(dates)} distinct dates, {dates[0]} -> {dates[-1]}.")
        updated = db.update_market_changes(_market_changes(dates))
        logger.info(f"Updated {updated} rows in {path.name}.")
    finally:
        db.close()


def main():
    """Backfill the S&P 500 market-change columns into every dataset in data/."""
    for path in _datasets_to_process():
        _backfill(path)


if __name__ == "__main__":
    main()
