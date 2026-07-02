"""Constants for the news + price-change dataset pipeline."""

import re
from pathlib import Path

# --- Finnhub company-news API -------------------------------------------------

# Finnhub company-news endpoint.
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"

# Name of the environment variable holding the (secret) Finnhub API key.
FINNHUB_API_KEY_ENV = "FINNHUB_API_KEY"

# Base delay (seconds) between retries after a transient Finnhub error, scaled up
# on each attempt.
FINNHUB_RATE_LIMIT_DELAY_SECONDS = 1.0

# Per-request timeout (seconds) for a Finnhub call.
FINNHUB_TIMEOUT_SECONDS = 20.0

# How many times a failing Finnhub request is retried before giving up.
FINNHUB_MAX_RETRIES = 3

# Width (in days) of each window the news date range is split into, so no single
# call silently drops the older days behind Finnhub's per-call result cap.
FINNHUB_CHUNK_DAYS = 7

# Maximum number of articles Finnhub returns per call; a window that hits this is
# split in half and re-fetched to guarantee full daily coverage.
FINNHUB_PAGE_CAP = 250

# Finnhub's free tier allows roughly this many requests per minute; calls are
# spaced out to stay safely under it.
FINNHUB_REQUESTS_PER_MINUTE = 50

# Minimum spacing (seconds) enforced between two Finnhub calls, derived from the
# per-minute allowance above.
FINNHUB_MIN_INTERVAL_SECONDS = 60.0 / max(1, FINNHUB_REQUESTS_PER_MINUTE)

# --- Symbols and market benchmark ---------------------------------------------

# Some companies file news under a legacy symbol on Finnhub. We query news with
# the alias but keep the canonical ticker for storage and price lookups.
SYMBOL_ALIASES = {
    "FI": "FISV",
}

# Symbol used as the broad-market benchmark whose return is subtracted from a
# stock's raw price change to isolate the news-driven (abnormal) move.
MARKET_INDEX_SYMBOL = "^GSPC"

# --- Volatility and price windows ---------------------------------------------

# Trailing window (in trading days) over which a stock's own daily-return
# volatility is estimated, used to standardize price moves into z-scores.
VOLATILITY_WINDOW_DAYS = 30

# Minimum number of daily returns required before a volatility estimate is
# considered reliable; fewer than this yields no estimate (None).
MIN_VOLATILITY_RETURNS = 20

# Calendar-day buffer fetched *before* the news range so the trailing volatility
# window has enough history even for the earliest news dates (30 trading days is
# roughly six weeks; extra slack covers weekends and holidays).
ESTIMATION_BUFFER_DAYS = 60

# Calendar-day buffer fetched *after* the news range so a 7-day forward change
# near the end of the range can still be computed.
FORWARD_BUFFER_DAYS = 14

# --- HTML / text cleaning -----------------------------------------------------

# Finnhub summaries occasionally contain raw HTML, encoded HTML, and even
# malformed tag remnants (e.g. a broken anchor whose opening "<a" was dropped,
# leaving `... href=... rel=noopener&gt;` at the start of the text).
REAL_TAG_RE = re.compile(r"<[^>]*>")
ENCODED_TAG_RE = re.compile(r"&lt;.*?&gt;", re.DOTALL)
BROKEN_LEAD_RE = re.compile(
    r"^\s*[^<>]*?(?:href|data-[\w-]+|rel|target|class|src|style)\s*=.*?(?:&gt;|>)",
    re.IGNORECASE | re.DOTALL,
)
WS_RE = re.compile(r"\s+")

# --- SQLite storage -----------------------------------------------------------

# Directory the built dataset databases are written to.
DATASET_OUTPUT_DIR = Path("data")

# Schema for the single news table every dataset database holds.
NEWS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    news_id              TEXT,
    news_title           TEXT NOT NULL,
    news_description     TEXT,
    ticker               TEXT NOT NULL,
    news_date            TEXT NOT NULL,
    price_change_1_day   REAL,
    price_change_7_days  REAL,
    market_change_1_day  REAL,
    market_change_7_days REAL,
    volatility_30_day    REAL,
    UNIQUE (ticker, news_date)
);
"""

# Columns added after the original schema; older databases are migrated lazily
# and readers treat them as optional.
OPTIONAL_COLUMNS = ("market_change_1_day", "market_change_7_days", "volatility_30_day")

# Columns selected when sampling news rows from a built dataset database.
SAMPLED_NEWS_COLUMNS = (
    "id, news_id, news_title, news_description, ticker, news_date, "
    "price_change_1_day, price_change_7_days"
)
