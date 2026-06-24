import sqlite3
from collections.abc import Iterable
from pathlib import Path

from src.models.news_record import NewsRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    news_id             TEXT,
    news_title          TEXT NOT NULL,
    news_description    TEXT,
    ticker              TEXT NOT NULL,
    news_date           TEXT NOT NULL,
    price_change_1_day  REAL,
    price_change_7_days REAL,
    UNIQUE (ticker, news_date)
);
"""


class NewsDatabase:
    """Thin SQLite wrapper that creates the schema and stores news rows."""

    def __init__(self, path: str | Path):
        self.__path = Path(path)
        self.__path.parent.mkdir(parents=True, exist_ok=True)
        self.__conn = sqlite3.connect(str(self.__path))
        self.__conn.row_factory = sqlite3.Row
        self.__create_schema()

    def __create_schema(self):
        self.__conn.executescript(_SCHEMA)
        self.__conn.commit()

    def upsert_many(self, records: Iterable[NewsRecord]) -> int:
        """Insert many records, replacing any existing row for the same ticker/day.

        Args:
            records (Iterable[NewsRecord]): Records to write.

        Returns:
            int: The number of rows written.
        """
        rows = [
            (
                r.news_id,
                r.news_title,
                r.news_description,
                r.ticker,
                r.news_date.isoformat(),
                r.price_change_1_day,
                r.price_change_7_days,
            )
            for r in records
        ]
        if not rows:
            return 0
        self.__conn.executemany(
            """
            INSERT INTO news (
                news_id, news_title, news_description, ticker, news_date,
                price_change_1_day, price_change_7_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (ticker, news_date) DO UPDATE SET
                news_id             = excluded.news_id,
                news_title          = excluded.news_title,
                news_description    = excluded.news_description,
                price_change_1_day  = excluded.price_change_1_day,
                price_change_7_days = excluded.price_change_7_days
            """,
            rows,
        )
        self.__conn.commit()
        return len(rows)

    def close(self):
        """Close the underlying SQLite connection."""
        self.__conn.close()

