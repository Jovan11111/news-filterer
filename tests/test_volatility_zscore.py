from datetime import date, timedelta
from math import sqrt

from src.constants.dataset import MIN_VOLATILITY_RETURNS
from src.dataset.news_sampler import NewsSampler
from src.dataset.price_fetcher import PriceFetcher
from src.evaluation.news_scorer import NewsScorer
from src.models.news_record import NewsRecord
from src.models.sampled_news import SampledNews
from src.utils.news_database import NewsDatabase
from src.utils.enums.filter_mode import FilterMode
from src.utils.enums.price_horizon import PriceHorizon


def make_article(price_1d=0.0, price_7d=0.0, market_1d=None, market_7d=None, volatility=None):
    return SampledNews(
        id=1,
        news_id="n1",
        news_title="title",
        news_description="",
        ticker="T1",
        news_date=date(2026, 1, 1),
        price_change_1_day=price_1d,
        price_change_7_days=price_7d,
        market_change_1_day=market_1d,
        market_change_7_days=market_7d,
        volatility_30_day=volatility,
    )


def inject_closes(fetcher, ticker, closes_by_date):
    # Seed the fetcher's private price cache directly (no network) for testing.
    fetcher._PriceFetcher__prices[ticker] = closes_by_date


def test_volatility_is_zero_for_constant_growth_series():
    fetcher = PriceFetcher()
    start = date(2025, 12, 1)
    # Each day grows by exactly 1% so every daily return is identical -> std 0.
    closes = {start + timedelta(days=i): 100.0 * (1.01**i) for i in range(25)}
    inject_closes(fetcher, "AAA", closes)
    news_date = start + timedelta(days=25)
    assert abs(fetcher.volatility("AAA", news_date)) < 1e-9


def test_volatility_ignores_the_news_day_close():
    fetcher = PriceFetcher()
    start = date(2025, 12, 1)
    closes = {start + timedelta(days=i): 100.0 * (1.01**i) for i in range(25)}
    news_date = start + timedelta(days=25)
    # A wild move ON the news day must not leak into the trailing estimate.
    closes[news_date] = 10_000.0
    inject_closes(fetcher, "AAA", closes)
    assert abs(fetcher.volatility("AAA", news_date)) < 1e-9


def test_volatility_none_when_insufficient_history():
    fetcher = PriceFetcher()
    start = date(2025, 12, 1)
    closes = {start + timedelta(days=i): 100.0 + i for i in range(MIN_VOLATILITY_RETURNS - 5)}
    inject_closes(fetcher, "AAA", closes)
    assert fetcher.volatility("AAA", date(2026, 1, 1)) is None


def test_volatility_none_for_unknown_ticker():
    fetcher = PriceFetcher()
    assert fetcher.volatility("MISSING", date(2026, 1, 1)) is None


def test_standardize_divides_one_day_move_by_volatility():
    scorer = NewsScorer(standardize=True)
    article = make_article(price_1d=2.0, volatility=4.0)
    assert scorer.price_change(article, PriceHorizon.ONE_DAY) == 0.5


def test_standardize_seven_day_scales_by_sqrt_five():
    scorer = NewsScorer(standardize=True)
    article = make_article(price_7d=5.0, volatility=2.0)
    expected = 5.0 / (2.0 * sqrt(5))
    assert scorer.price_change(article, PriceHorizon.SEVEN_DAYS) == expected


def test_standardize_stacks_with_market_adjustment():
    scorer = NewsScorer(subtract_market=True, standardize=True)
    article = make_article(price_1d=3.0, market_1d=1.0, volatility=4.0)
    # (3 - 1) / 4 = 0.5
    assert scorer.price_change(article, PriceHorizon.ONE_DAY) == 0.5


def test_standardize_drops_articles_without_volatility():
    scorer = NewsScorer(standardize=True)
    assert scorer.price_change(make_article(price_1d=2.0, volatility=None), PriceHorizon.ONE_DAY) is None
    assert scorer.price_change(make_article(price_1d=2.0, volatility=0.0), PriceHorizon.ONE_DAY) is None


def test_standardize_changes_ranking_toward_low_volatility_movers():
    scorer = NewsScorer(standardize=True)
    calm = make_article(price_1d=2.0, volatility=1.0)   # 2 sigma
    jumpy = make_article(price_1d=3.0, volatility=6.0)  # 0.5 sigma
    pairs = scorer.oriented_pairs([jumpy, calm], FilterMode.BUY, PriceHorizon.ONE_DAY)
    assert pairs[0][1] == 0.5
    assert pairs[1][1] == 2.0


def test_database_roundtrips_volatility(tmp_path):
    db_path = tmp_path.joinpath("vol.db")
    db = NewsDatabase(db_path)
    db.upsert_many([
        NewsRecord(
            news_id="AAA-1",
            news_title="AAA headline",
            news_description="d",
            ticker="AAA",
            news_date=date(2026, 1, 2),
            price_change_1_day=1.0,
            price_change_7_days=2.0,
            market_change_1_day=0.3,
            market_change_7_days=0.4,
            volatility_30_day=1.75,
        )
    ])
    db.close()

    rows = NewsSampler(database=db_path, batch_size=10, min_gap_days=0).sample()
    assert len(rows) == 1
    assert rows[0].volatility_30_day == 1.75
