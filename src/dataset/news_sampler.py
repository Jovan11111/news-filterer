import random
import sqlite3
from datetime import date
from pathlib import Path

from src.constants.dataset import OPTIONAL_COLUMNS, SAMPLED_NEWS_COLUMNS
from src.models.sampled_news import SampledNews
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NewsSampler:
    """Draws random, per-ticker time-spaced batches of news from one database."""

    def __init__(
        self,
        database: Path,
        batch_size: int = 50,
        min_gap_days: int = 7,
        seed: int = 74,
    ):
        """Create the sampler for one database and sampling constraints.

        Args:
            database (Path): Path to the dataset database to sample from.
            batch_size (int): Number of rows to draw per sample.
            min_gap_days (int): Minimum days between two picks of the same ticker.
            seed (int): Seed for the internal random generator.
        """
        self.__db_path = database
        self.__batch_size = batch_size
        self.__min_gap_days = min_gap_days
        self.__rng = random.Random(seed)

    @property
    def batch_size(self) -> int:
        """The number of rows drawn per sample."""
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        """Set the number of rows drawn per sample."""
        self.__batch_size = value

    @property
    def db_path(self) -> Path:
        """The path to the database being sampled."""
        return self.__db_path

    @db_path.setter
    def db_path(self, value: Path):
        """Set the path to the database being sampled."""
        self.__db_path = value

    def seed(self, value: int):
        """Reseed the random generator used for sampling.

        Args:
            value (int): The new seed for the random generator.
        """
        self.__rng = random.Random(value)

    def __load_all(self) -> list[SampledNews]:
        """Load every news row from the database into memory."""
        conn = sqlite3.connect(str(self.__db_path))
        conn.row_factory = sqlite3.Row
        try:
            present = self.__present_optional_columns(conn)
            columns = SAMPLED_NEWS_COLUMNS + "".join(f", {column}" for column in present)
            cur = conn.execute(f"SELECT {columns} FROM news")
            return [
                SampledNews(
                    id=row["id"],
                    news_id=row["news_id"],
                    news_title=row["news_title"],
                    news_description=row["news_description"],
                    ticker=row["ticker"],
                    news_date=date.fromisoformat(row["news_date"]),
                    price_change_1_day=row["price_change_1_day"],
                    price_change_7_days=row["price_change_7_days"],
                    market_change_1_day=row["market_change_1_day"] if "market_change_1_day" in present else None,
                    market_change_7_days=row["market_change_7_days"] if "market_change_7_days" in present else None,
                    volatility_30_day=row["volatility_30_day"] if "volatility_30_day" in present else None,
                )
                for row in cur
            ]
        finally:
            conn.close()

    @staticmethod
    def __present_optional_columns(conn: sqlite3.Connection) -> set[str]:
        """Return which of the optional (later-added) columns the table carries."""
        names = {row["name"] for row in conn.execute("PRAGMA table_info(news)")}
        return set(OPTIONAL_COLUMNS) & names

    def sample(self) -> list[SampledNews]:
        """Draw a random batch of news rows subject to the per-ticker time gap.

        Returns:
            list[SampledNews]: Chosen rows, each spaced by at least min_gap_days per ticker.
        """
        rows = self.__load_all()
        self.__rng.shuffle(rows)

        selected: list[SampledNews] = []
        per_ticker_dates: dict[str, list[date]] = {}
        for row in rows:
            if len(selected) >= self.__batch_size:
                break
            row_date = row.news_date
            chosen = per_ticker_dates.get(row.ticker)
            if chosen is not None and any(
                abs((row_date - d).days) < self.__min_gap_days for d in chosen
            ):
                continue
            selected.append(row)
            per_ticker_dates.setdefault(row.ticker, []).append(row_date)

        if len(selected) < self.__batch_size:
            logger.warning(
                f"Requested {self.__batch_size} rows but only {len(selected)} fit the {self.__min_gap_days}-day per-ticker gap in {self.__db_path.name}."
            )
        return selected
