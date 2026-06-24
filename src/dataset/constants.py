import re

# Finnhub company-news endpoint.
BASE_URL = "https://finnhub.io/api/v1/company-news"

# Some companies file news under a legacy symbol on Finnhub. We query news with
# the alias but keep the canonical ticker for storage and price lookups.
SYMBOL_ALIASES = {
    "FI": "FISV",
}

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
