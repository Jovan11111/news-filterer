from datetime import date
from pathlib import Path

from src.constants.dataset import MARKET_INDEX_SYMBOL
from src.dataset.finnhub_news_fetcher import FinnhubNewsFetcher
from src.dataset.price_fetcher import PriceFetcher
from src.models.news_record import NewsRecord
from src.utils.config_provider import ConfigProvider
from src.utils.news_database import NewsDatabase
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetBuilder:
    """Builds a news + price-change dataset and persists it to SQLite."""

    def __init__(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
    ):
        """Create the builder for a ticker list and date range.

        Args:
            tickers (list[str]): The tickers to collect news and prices for.
            start_date (date): First date (inclusive) of the news range.
            end_date (date): Last date (inclusive) of the news range.

        Raises:
            ValueError: If start_date is after end_date.
        """
        self.__tickers = tickers
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) must not be after end_date ({end_date})."
            )
        self.__start = start_date
        self.__end = end_date

        self.__news_fetcher = FinnhubNewsFetcher(start_date, end_date)
        self.__price_fetcher = PriceFetcher()

    @property
    def db_path(self) -> Path:
        """Path of the DB that will be created, name of the db is: sp{N}_{start}_{end}."""
        name = f"sp{len(self.__tickers)}_{self.__start.isoformat()}_{self.__end.isoformat()}.db"
        return ConfigProvider.get().dataset_output_dir.joinpath(name)

    def build(self) -> Path:
        """Run the full pipeline and return the path of the written database.

        Returns:
            Path: Path to the database file containing the news + price-change dataset.
        """
        logger.info(
            f"Building dataset: {len(self.__tickers)} tickers, {self.__start} -> {self.__end}"
        )
        db = NewsDatabase(self.db_path)
        try:
            self.__price_fetcher.load_history(MARKET_INDEX_SYMBOL, self.__start, self.__end)
            for idx, ticker in enumerate(self.__tickers, start=1):
                self.__process_ticker(db, ticker, idx)
        finally:
            db.close()
        return self.db_path

    def __process_ticker(self, db: NewsDatabase, ticker: str, idx: int):
        """Fetch, price and persist the news rows for a single ticker."""
        try:
            per_day = self.__news_fetcher.fetch_one_per_day(ticker)
        except Exception as e:
            logger.warning(f"News fetch failed for {ticker}: {e}")
            return

        if not per_day:
            logger.info(f"No news found for {ticker}")
            return

        self.__price_fetcher.load_history(ticker, self.__start, self.__end)
        if not self.__price_fetcher.has_history(ticker):
            logger.warning(f"No price history for {ticker}; skipping its {len(per_day)} news rows.")
            return

        records: list[NewsRecord] = []
        for news_date, item in sorted(per_day.items()):
            price = self.__price_fetcher.price(ticker, news_date)
            market = self.__price_fetcher.price(MARKET_INDEX_SYMBOL, news_date)
            volatility = self.__price_fetcher.volatility(ticker, news_date)
            records.append(
                NewsRecord(
                    news_id=item.news_id,
                    news_title=item.title,
                    news_description=item.description,
                    ticker=ticker,
                    news_date=news_date,
                    price_change_1_day=price.change_1_day,
                    price_change_7_days=price.change_7_days,
                    market_change_1_day=market.change_1_day,
                    market_change_7_days=market.change_7_days,
                    volatility_30_day=volatility,
                )
            )

        db.upsert_many(records)
