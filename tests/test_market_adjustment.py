import sqlite3
from datetime import date

from src.dataset.news_sampler import NewsSampler
from src.evaluation.news_scorer import NewsScorer
from src.models.news_record import NewsRecord
from src.models.sampled_news import SampledNews
from src.utils.news_database import NewsDatabase
from src.utils.enums.filter_mode import FilterMode
from src.utils.enums.price_horizon import PriceHorizon


def make_article(article_id, price_1d, market_1d=None, price_7d=None, market_7d=None):
    return SampledNews(
        id=article_id,
        news_id=f"n{article_id}",
        news_title=f"title {article_id}",
        news_description="",
        ticker=f"T{article_id}",
        news_date=date(2026, 1, 1),
        price_change_1_day=price_1d,
        price_change_7_days=price_7d,
        market_change_1_day=market_1d,
        market_change_7_days=market_7d,
    )


def make_record(ticker, news_date, price_1d, price_7d, market_1d=None, market_7d=None):
    return NewsRecord(
        news_id=f"{ticker}-{news_date}",
        news_title=f"{ticker} headline",
        news_description="desc",
        ticker=ticker,
        news_date=news_date,
        price_change_1_day=price_1d,
        price_change_7_days=price_7d,
        market_change_1_day=market_1d,
        market_change_7_days=market_7d,
    )


def test_raw_scorer_ignores_market():
    scorer = NewsScorer(subtract_market=False)
    article = make_article(1, price_1d=2.5, market_1d=2.0)
    assert scorer.price_change(article, PriceHorizon.ONE_DAY) == 2.5


def test_market_adjusted_scorer_subtracts_market():
    scorer = NewsScorer(subtract_market=True)
    article = make_article(1, price_1d=2.5, market_1d=2.0)
    assert scorer.price_change(article, PriceHorizon.ONE_DAY) == 0.5


def test_market_adjusted_falls_back_to_raw_when_market_missing():
    scorer = NewsScorer(subtract_market=True)
    article = make_article(1, price_1d=2.5, market_1d=None)
    assert scorer.price_change(article, PriceHorizon.ONE_DAY) == 2.5


def test_market_adjusted_seven_day_horizon():
    scorer = NewsScorer(subtract_market=True)
    article = make_article(1, price_1d=0.0, market_1d=0.0, price_7d=5.0, market_7d=3.0)
    assert scorer.price_change(article, PriceHorizon.SEVEN_DAYS) == 2.0


def test_adjustment_changes_buy_ordering():
    raw = NewsScorer(subtract_market=False)
    adjusted = NewsScorer(subtract_market=True)
    # Article A rode the market; B beat a falling market.
    a = make_article(1, price_1d=3.0, market_1d=3.0)
    b = make_article(2, price_1d=1.0, market_1d=-1.0)
    raw_pairs = raw.oriented_pairs([a, b], FilterMode.BUY, PriceHorizon.ONE_DAY)
    adj_pairs = adjusted.oriented_pairs([a, b], FilterMode.BUY, PriceHorizon.ONE_DAY)
    assert [value for _, value in raw_pairs] == [3.0, 1.0]
    assert [value for _, value in adj_pairs] == [0.0, 2.0]


def test_database_roundtrips_market_columns(tmp_path):
    db_path = tmp_path.joinpath("sample.db")
    db = NewsDatabase(db_path)
    db.upsert_many([make_record("AAA", date(2026, 1, 2), 1.0, 2.0, 0.3, 0.4)])
    db.close()

    rows = NewsSampler(database=db_path, batch_size=10, min_gap_days=0).sample()
    assert len(rows) == 1
    assert rows[0].market_change_1_day == 0.3
    assert rows[0].market_change_7_days == 0.4


def test_backfill_updates_existing_rows(tmp_path):
    db_path = tmp_path.joinpath("backfill.db")
    db = NewsDatabase(db_path)
    db.upsert_many([
        make_record("AAA", date(2026, 1, 2), 1.0, 2.0),
        make_record("BBB", date(2026, 1, 2), 1.5, 2.5),
        make_record("CCC", date(2026, 1, 9), -1.0, -2.0),
    ])
    assert db.distinct_news_dates() == [date(2026, 1, 2), date(2026, 1, 9)]

    updated = db.update_market_changes({
        date(2026, 1, 2): (0.5, 1.0),
        date(2026, 1, 9): (-0.5, -1.0),
    })
    db.close()
    assert updated == 3

    rows = sorted(NewsSampler(database=db_path, batch_size=10, min_gap_days=0).sample(), key=lambda r: r.ticker)
    assert rows[0].market_change_1_day == 0.5
    assert rows[2].market_change_1_day == -0.5


def test_migration_adds_columns_to_legacy_database(tmp_path):
    db_path = tmp_path.joinpath("legacy.db")
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(
        """
        CREATE TABLE news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_id TEXT, news_title TEXT NOT NULL, news_description TEXT,
            ticker TEXT NOT NULL, news_date TEXT NOT NULL,
            price_change_1_day REAL, price_change_7_days REAL,
            UNIQUE (ticker, news_date)
        );
        INSERT INTO news (news_id, news_title, news_description, ticker, news_date, price_change_1_day, price_change_7_days)
        VALUES ('x', 'old headline', 'd', 'AAA', '2026-01-02', 1.0, 2.0);
        """
    )
    legacy.commit()
    legacy.close()

    # A legacy DB has no market columns; the sampler must still load it.
    rows = NewsSampler(database=db_path, batch_size=10, min_gap_days=0).sample()
    assert rows[0].market_change_1_day is None

    # Opening through NewsDatabase migrates the schema in place.
    NewsDatabase(db_path).close()
    columns = {row[1] for row in sqlite3.connect(str(db_path)).execute("PRAGMA table_info(news)")}
    assert {"market_change_1_day", "market_change_7_days"} <= columns
