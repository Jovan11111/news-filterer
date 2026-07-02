import time
from datetime import date, datetime, timedelta, timezone

import requests

from src.constants.dataset import (
    FINNHUB_API_KEY_ENV,
    FINNHUB_CHUNK_DAYS,
    FINNHUB_MAX_RETRIES,
    FINNHUB_MIN_INTERVAL_SECONDS,
    FINNHUB_NEWS_URL,
    FINNHUB_PAGE_CAP,
    FINNHUB_RATE_LIMIT_DELAY_SECONDS,
    FINNHUB_TIMEOUT_SECONDS,
    SYMBOL_ALIASES,
)
from src.models.news_item import NewsItem
from src.utils.clean_text import clean_text
from src.utils.config_provider import ConfigProvider
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Design decisions for fetching company news from Finnhub
# -------------------------------------------------------
#
# 1. Why we split the date range into windows (chunk_days)
#    Finnhub returns at most ~page_cap (~250) articles per single call and gives
#    back the *newest* ones first. So one wide-range request for an active ticker
#    silently drops the older days. We instead request the range in small
#    chunk_days-sized windows so every day is actually covered.
#
# 2. Why windows split themselves further (adaptive sub-chunking)
#    Even a small window can hit the ~250 cap for a very busy ticker, which would
#    again drop days. So whenever a window comes back full and still spans more
#    than one day, we cut it in half and fetch each half (recursively). This
#    guarantees full daily coverage even for the busiest symbols, without making
#    tiny one-day calls everywhere when they are not needed.
#
# 3. Why we throttle *before* sending (proactive rate limiting)
#    Finnhub's free tier allows ~60 requests/min. Spacing our calls out by
#    min_interval keeps us safely under that limit and avoids 429s almost
#    entirely, which is far more reliable than firing fast and reacting to 429s
#    after the fact. The 429/error handling below is only a safety net.
#
# 4. Why one article per day (fetch_one_per_day)
#    The dataset is one row per (ticker, day), so we need a single article per
#    day. When a day has several, we keep the one with the longest summary, as a
#    simple proxy for "most informative" for downstream ranking.
#
# 5. Why symbol aliases and de-duplication
#    A few companies are indexed under a legacy symbol on Finnhub (SYMBOL_ALIASES),
#    so we query with the alias but keep the canonical ticker. Articles are also
#    de-duplicated by their Finnhub id, because overlapping windows can return the
#    same article more than once.


class FinnhubNewsFetcher:
    """Collects real company news for a ticker over an explicit date range."""

    def __init__(self, start: date, end: date):
        """Create the fetcher for a date range and open an HTTP session.

        Args:
            start (date): First date (inclusive) of news to fetch.
            end (date): Last date (inclusive) of news to fetch.
        """
        self.__api_key = ConfigProvider.get().finnhub_api_key
        if not self.__api_key:
            raise ValueError(f"Missing required environment variable: {FINNHUB_API_KEY_ENV}")
        self.__start = start
        self.__end = end
        self.__last_request_ts = 0.0
        self.__session = requests.Session()

    def __throttle(self):
        """Block until at least min_interval has passed since the last call."""
        elapsed = time.monotonic() - self.__last_request_ts
        if elapsed < FINNHUB_MIN_INTERVAL_SECONDS:
            time.sleep(FINNHUB_MIN_INTERVAL_SECONDS - elapsed)
        self.__last_request_ts = time.monotonic()

    def __request(self, ticker: str, start: date, end: date) -> list[dict]:
        """Call the Finnhub company-news endpoint with throttling and retries, returning raw dicts."""
        params = {
            "symbol": SYMBOL_ALIASES.get(ticker, ticker),
            "from": start.isoformat(),
            "to": end.isoformat(),
            "token": self.__api_key,
        }
        for attempt in range(1, FINNHUB_MAX_RETRIES + 1):
            self.__throttle()
            try:
                resp = self.__session.get(FINNHUB_NEWS_URL, params=params, timeout=FINNHUB_TIMEOUT_SECONDS)
            except requests.RequestException as e:
                logger.warning(
                    f"Finnhub request error for {ticker} (try {attempt}): {e}"
                )
                time.sleep(FINNHUB_RATE_LIMIT_DELAY_SECONDS * attempt)
                continue

            if resp.status_code == 429:  # too many requests, need to slow down
                wait = FINNHUB_RATE_LIMIT_DELAY_SECONDS * attempt * 2
                logger.warning(f"Finnhub rate limit for {ticker}; sleeping {wait:.1f}s")
                time.sleep(wait)
                continue
            if resp.status_code == 401:
                raise RuntimeError(
                    "Finnhub rejected the API key (401). Check FINNHUB_API_KEY."
                )
            if resp.status_code != 200:
                logger.warning(f"Finnhub HTTP {resp.status_code} for {ticker}")
                time.sleep(FINNHUB_RATE_LIMIT_DELAY_SECONDS * attempt)
                continue

            data = resp.json()
            return data if isinstance(data, list) else []
        return []

    def fetch(self, ticker: str) -> list[NewsItem]:
        """Fetch articles for a specified ticker in the configured time range.

        Args:
            ticker (str): The ticker symbol of the company.

        Returns:
            list[NewsItem]: A list of news items for the specified ticker and date range.
        """
        items: dict[str, NewsItem] = {}
        for raw in self.__request_range(ticker):
            ts = raw.get("datetime")
            if not ts:
                continue
            pub = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
            if not (self.__start <= pub <= self.__end):
                continue
            title = clean_text(raw.get("headline") or "")
            if not title:
                continue
            news_id = str(raw.get("id") or raw.get("url") or title)
            items[news_id] = NewsItem(
                news_id=news_id,
                title=title,
                description=clean_text(raw.get("summary") or ""),
                ticker=ticker,
                pub_date=pub,
            )
        return list(items.values())

    def __request_range(self, ticker: str) -> list[dict]:
        """Fetch the full range with adaptive sub-chunking, concatenated."""
        raw: list[dict] = []
        window_start = self.__start
        while window_start <= self.__end:
            window_end = min(window_start + timedelta(days=FINNHUB_CHUNK_DAYS - 1), self.__end)
            raw.extend(self.__fetch_window(ticker, window_start, window_end))
            window_start = window_end + timedelta(days=1)
        return raw

    def __fetch_window(self, ticker: str, start: date, end: date) -> list[dict]:
        """Fetch one window, recursively splitting if it hits the result cap."""
        items = self.__request(ticker, start, end)
        if len(items) >= FINNHUB_PAGE_CAP and start < end:
            mid = start + timedelta(days=(end - start).days // 2)
            left = self.__fetch_window(ticker, start, mid)
            right = self.__fetch_window(ticker, mid + timedelta(days=1), end)
            return left + right
        return items

    def fetch_one_per_day(self, ticker: str) -> dict[date, NewsItem]:
        """Return a single article per day, so a ticker never has two same-day rows.

        When a day has several articles, the one with the longest summary is kept.

        Args:
            ticker (str): The ticker symbol of the company.

        Returns:
            dict[date, NewsItem]: One news item per publication day.
        """
        best: dict[date, NewsItem] = {}
        for item in self.fetch(ticker):
            current = best.get(item.pub_date)
            if current is None or len(item.description) > len(current.description):
                best[item.pub_date] = item
        return best
