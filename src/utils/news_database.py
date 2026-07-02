import sqlite3
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from src.models.news_record import NewsRecord


class NewsDatabase:
    """Thin SQLite wrapper that creates the schema and stores news rows."""

    __SCHEMA = """
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

    # Columns added after the original schema; older databases are migrated lazily.
    __ADDED_COLUMNS = ("market_change_1_day", "market_change_7_days", "volatility_30_day")

    def __init__(self, path: str | Path):
        """Open (creating if needed) the database and ensure the schema exists.

        Args:
            path (str | Path): Path to the SQLite database file.
        """
        self.__path = Path(path)
        self.__path.parent.mkdir(parents=True, exist_ok=True)
        self.__conn = sqlite3.connect(str(self.__path))
        self.__conn.row_factory = sqlite3.Row
        self.__create_schema()

    def __create_schema(self):
        """Create the news table if missing and migrate older databases."""
        self.__conn.executescript(self.__SCHEMA)
        self.__migrate()
        self.__conn.commit()

    def __migrate(self):
        """Add columns introduced after the original schema to older databases."""
        existing = {row["name"] for row in self.__conn.execute("PRAGMA table_info(news)")}
        for column in self.__ADDED_COLUMNS:
            if column not in existing:
                self.__conn.execute(f"ALTER TABLE news ADD COLUMN {column} REAL")

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
                r.market_change_1_day,
                r.market_change_7_days,
                r.volatility_30_day,
            )
            for r in records
        ]
        if not rows:
            return 0
        self.__conn.executemany(
            """
            INSERT INTO news (
                news_id, news_title, news_description, ticker, news_date,
                price_change_1_day, price_change_7_days,
                market_change_1_day, market_change_7_days, volatility_30_day
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (ticker, news_date) DO UPDATE SET
                news_id              = excluded.news_id,
                news_title           = excluded.news_title,
                news_description     = excluded.news_description,
                price_change_1_day   = excluded.price_change_1_day,
                price_change_7_days  = excluded.price_change_7_days,
                market_change_1_day  = excluded.market_change_1_day,
                market_change_7_days = excluded.market_change_7_days,
                volatility_30_day    = excluded.volatility_30_day
            """,
            rows,
        )
        self.__conn.commit()
        return len(rows)

    def distinct_news_dates(self) -> list[date]:
        """Return every distinct news date in the table, ascending.

        Returns:
            list[date]: The sorted distinct publication dates.
        """
        cursor = self.__conn.execute("SELECT DISTINCT news_date FROM news ORDER BY news_date")
        return [date.fromisoformat(row["news_date"]) for row in cursor]

    def update_market_changes(self, changes_by_date: dict[date, tuple[float | None, float | None]]) -> int:
        """Set the 1-/7-day market change for every row on each given news date.

        Args:
            changes_by_date (dict): Maps a news date to its (1-day, 7-day) market change.

        Returns:
            int: The number of rows updated.
        """
        updated = 0
        for news_date, (change_1_day, change_7_days) in changes_by_date.items():
            cursor = self.__conn.execute(
                "UPDATE news SET market_change_1_day = ?, market_change_7_days = ? WHERE news_date = ?",
                (change_1_day, change_7_days, news_date.isoformat()),
            )
            updated += cursor.rowcount
        self.__conn.commit()
        return updated

    def close(self):
        """Close the underlying SQLite connection."""
        self.__conn.close()
