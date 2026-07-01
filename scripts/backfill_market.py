import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date  # noqa: E402

from src.dataset.constants import MARKET_INDEX_SYMBOL  # noqa: E402
from src.dataset.price_fetcher import PriceFetcher  # noqa: E402
from src.utils.database import NewsDatabase  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.joinpath("data")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the market backfill."""
    parser = argparse.ArgumentParser(description="Backfill S&P 500 market change columns into existing datasets.")
    parser.add_argument("--dataset", dest="dataset", default=None, help="Path to a single .db to backfill (default: every .db in data/).")
    return parser.parse_args()


def datasets_to_process(dataset: str | None) -> list[Path]:
    """Resolve which database files the backfill should touch."""
    if dataset is not None:
        path = Path(dataset)
        if not path.exists():
            raise SystemExit(f"Dataset not found: {path}")
        return [path]
    found = sorted(DATA_DIR.glob("*.db"))
    if not found:
        raise SystemExit(f"No dataset databases found in {DATA_DIR}.")
    return found


def market_changes(dates: list[date]) -> dict[date, tuple[float | None, float | None]]:
    """Compute the 1-/7-day S&P 500 change for each news date."""
    fetcher = PriceFetcher()
    fetcher.load_history(MARKET_INDEX_SYMBOL, min(dates), max(dates))
    changes: dict[date, tuple[float | None, float | None]] = {}
    for news_date in dates:
        price = fetcher.price(MARKET_INDEX_SYMBOL, news_date)
        changes[news_date] = (price.change_1_day, price.change_7_days)
    return changes


def backfill(path: Path):
    """Add and populate the market-change columns for a single dataset."""
    db = NewsDatabase(path)
    try:
        dates = db.distinct_news_dates()
        if not dates:
            logger.warning(f"{path.name} has no news rows; skipping.")
            return
        logger.info(f"Backfilling {path.name}: {len(dates)} distinct dates, {dates[0]} -> {dates[-1]}.")
        updated = db.update_market_changes(market_changes(dates))
        logger.info(f"Updated {updated} rows in {path.name}.")
    finally:
        db.close()


def main():
    args = parse_args()
    for path in datasets_to_process(args.dataset):
        backfill(path)


if __name__ == "__main__":
    main()
