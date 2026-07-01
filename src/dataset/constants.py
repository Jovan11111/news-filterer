import re

# Finnhub company-news endpoint.
BASE_URL = "https://finnhub.io/api/v1/company-news"

# Some companies file news under a legacy symbol on Finnhub. We query news with
# the alias but keep the canonical ticker for storage and price lookups.
SYMBOL_ALIASES = {
    "FI": "FISV",
}

# Symbol used as the broad-market benchmark whose return is subtracted from a
# stock's raw price change to isolate the news-driven (abnormal) move.
MARKET_INDEX_SYMBOL = "^GSPC"

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

# Columns selected when sampling news rows from a built dataset database.
SAMPLED_NEWS_COLUMNS = (
    "id, news_id, news_title, news_description, ticker, news_date, "
    "price_change_1_day, price_change_7_days"
)
