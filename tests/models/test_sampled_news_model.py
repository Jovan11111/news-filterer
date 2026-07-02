from datetime import date

from src.models.sampled_news import SampledNews


def __make() -> SampledNews:
    return SampledNews(
        id=7,
        news_id="n1",
        news_title="Title",
        news_description="Description",
        ticker="AAPL",
        news_date=date(2026, 1, 2),
        price_change_1_day=1.5,
        price_change_7_days=2.5,
        market_change_1_day=0.3,
        market_change_7_days=0.4,
        volatility_30_day=1.75,
    )


def test_properties_return_stored_values():
    row = __make()
    assert row.id == 7
    assert row.news_id == "n1"
    assert row.news_title == "Title"
    assert row.news_description == "Description"
    assert row.ticker == "AAPL"
    assert row.news_date == date(2026, 1, 2)
    assert row.price_change_1_day == 1.5
    assert row.price_change_7_days == 2.5
    assert row.market_change_1_day == 0.3
    assert row.market_change_7_days == 0.4
    assert row.volatility_30_day == 1.75


def test_optional_fields_default_to_none():
    row = SampledNews(
        id=1,
        news_id="n2",
        news_title="T",
        news_description=None,
        ticker="MSFT",
        news_date=date(2026, 1, 3),
        price_change_1_day=None,
        price_change_7_days=None,
    )
    assert row.news_description is None
    assert row.market_change_1_day is None
    assert row.market_change_7_days is None
    assert row.volatility_30_day is None


def test_to_dict_returns_all_fields():
    assert __make().to_dict() == {
        "id": 7,
        "news_id": "n1",
        "news_title": "Title",
        "news_description": "Description",
        "ticker": "AAPL",
        "news_date": "2026-01-02",
        "price_change_1_day": 1.5,
        "price_change_7_days": 2.5,
        "market_change_1_day": 0.3,
        "market_change_7_days": 0.4,
        "volatility_30_day": 1.75,
    }
