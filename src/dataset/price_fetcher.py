from datetime import date, timedelta
from statistics import stdev

import yfinance as yf

from src.dataset.constants import (
    ESTIMATION_BUFFER_DAYS,
    MIN_VOLATILITY_RETURNS,
    SYMBOL_ALIASES,
    VOLATILITY_WINDOW_DAYS,
)
from src.models.price import Price
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PriceFetcher:
    """Computes 1-day and 7-day forward price changes for a ticker."""

    def __init__(self):
        """Create an empty fetcher with no loaded price history."""
        self.__prices: dict[str, dict[date, float]] = {}

    @staticmethod
    def __yahoo_symbol(ticker: str) -> str:
        """Translate a canonical ticker into the symbol Yahoo Finance expects."""
        return SYMBOL_ALIASES.get(ticker, ticker).replace(".", "-")

    def load_history(self, ticker: str, start: date, end: date):
        """Download daily closes for ticker covering [start, end] plus a buffer.

        A trailing buffer is added so 7-day forward changes near the end can
        still be computed, and a leading buffer (ESTIMATION_BUFFER_DAYS) so the
        trailing volatility window has enough history for the earliest news.

        Args:
            ticker (str): The ticker whose price history is downloaded.
            start (date): First day of news coverage.
            end (date): Last day of news coverage.
        """
        buf_start = start - timedelta(days=ESTIMATION_BUFFER_DAYS)
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

    def has_history(self, ticker: str) -> bool:
        """Return whether any daily closes were loaded for a ticker.

        Args:
            ticker (str): The ticker to check.

        Returns:
            bool: True if the ticker has loaded price history (is priceable).
        """
        return bool(self.__prices.get(ticker))

    @staticmethod
    def __close_on_or_before(closes: dict[date, float], target: date) -> float | None:
        """Return the close on or before the target date, or None when none exists."""
        prior = [d for d in closes if d <= target]
        return closes[max(prior)] if prior else None

    @staticmethod
    def __close_on_or_after(closes: dict[date, float], target: date) -> float | None:
        """Return the close on or after the target date, or None when none exists."""
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

    def volatility(self, ticker: str, news_date: date, window: int = VOLATILITY_WINDOW_DAYS) -> float | None:
        """Trailing daily-return volatility (%) of a ticker as of a news date.

        The standard deviation of the stock's daily simple returns over the last
        ``window`` trading days strictly *before* ``news_date`` (so the news-day
        move itself never leaks into the estimate). Returns ``None`` when too
        little history is available to be reliable.

        Args:
            ticker (str): The ticker whose volatility is estimated.
            news_date (date): The day whose prior history defines the window.
            window (int): Number of trailing trading days to include.

        Returns:
            float | None: The daily-return standard deviation in percent, or ``None``.
        """
        closes = self.__prices.get(ticker)
        if not closes:
            return None
        prior_dates = sorted(day for day in closes if day < news_date)
        recent = prior_dates[-(window + 1):]
        returns = []
        for previous, current in zip(recent, recent[1:]):
            base = closes[previous]
            if base:
                returns.append((closes[current] - base) / base * 100.0)
        if len(returns) < MIN_VOLATILITY_RETURNS:
            return None
        return round(stdev(returns), 6)
