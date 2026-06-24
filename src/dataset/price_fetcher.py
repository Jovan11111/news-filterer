from datetime import date, timedelta

import yfinance as yf

from src.dataset.constants import SYMBOL_ALIASES
from src.models.price import Price
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PriceFetcher:
    """Computes 1-day and 7-day forward price changes for a ticker."""

    def __init__(self):
        self.__prices: dict[str, dict[date, float]] = {}

    @staticmethod
    def __yahoo_symbol(ticker: str) -> str:
        """Translate a canonical ticker into the symbol Yahoo Finance expects."""
        return SYMBOL_ALIASES.get(ticker, ticker).replace(".", "-")

    def load_history(self, ticker: str, start: date, end: date):
        """Download daily closes for ticker covering [start, end] plus a buffer.

        A trailing buffer is added so 7-day forward changes near the end can
        still be computed, and a small leading buffer so a base price exists for
        news published on the very first day.

        Args:
            ticker (str): The ticker whose price history is downloaded.
            start (date): First day of news coverage.
            end (date): Last day of news coverage.
        """
        buf_start = start - timedelta(days=7)
        buf_end = end + timedelta(days=14)
        try:
            hist = yf.Ticker(self.__yahoo_symbol(ticker)).history(
                start=buf_start.isoformat(),
                end=buf_end.isoformat(),
                interval="1d",
                auto_adjust=True,
            )
        except Exception as e:
            logger.warning(f"history failed for {ticker}: {e}")
            self.__prices[ticker] = {}
            return

        if hist.empty or "Close" not in hist.columns:
            self.__prices[ticker] = {}
            return

        self.__prices[ticker] = {ts.date(): float(close) for ts, close in hist["Close"].items()}

    @staticmethod
    def __close_on_or_before(closes: dict[date, float], target: date) -> float | None:
        prior = [d for d in closes if d <= target]
        return closes[max(prior)] if prior else None

    @staticmethod
    def __close_on_or_after(closes: dict[date, float], target: date) -> float | None:
        later = [d for d in closes if d >= target]
        return closes[min(later)] if later else None

    def price(self, ticker: str, news_date: date) -> Price:
        """Build the base and 1-/7-day forward closing prices for a news date.

        The base is the close on or before the news date; the forward prices are
        the first available closes on or after the news date + 1 and + 7 days.

        Args:
            ticker (str): The ticker the news refers to.
            news_date (date): The day the news was published.

        Returns:
            Price: The base and forward closing prices for the ticker.
        """
        closes = self.__prices.get(ticker)
        if closes is None:
            self.load_history(ticker, news_date, news_date)
            closes = self.__prices[ticker]

        base = self.__close_on_or_before(closes, news_date)
        close_1d = self.__close_on_or_after(closes, news_date + timedelta(days=1))
        close_7d = self.__close_on_or_after(closes, news_date + timedelta(days=7))
        return Price(ticker, base, close_1d, close_7d)
