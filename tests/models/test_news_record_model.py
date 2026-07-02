from datetime import date

from src.models.news_record import NewsRecord


def __make() -> NewsRecord:
    return NewsRecord(
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
    record = __make()
    assert record.news_id == "n1"
    assert record.news_title == "Title"
    assert record.news_description == "Description"
    assert record.ticker == "AAPL"
    assert record.news_date == date(2026, 1, 2)
    assert record.price_change_1_day == 1.5
    assert record.price_change_7_days == 2.5
    assert record.market_change_1_day == 0.3
    assert record.market_change_7_days == 0.4
    assert record.volatility_30_day == 1.75


def test_optional_fields_default_to_none():
    record = NewsRecord(
        news_id="n2",
        news_title="T",
        news_description="D",
        ticker="MSFT",
        news_date=date(2026, 1, 3),
        price_change_1_day=None,
        price_change_7_days=None,
    )
    assert record.market_change_1_day is None
    assert record.market_change_7_days is None
    assert record.volatility_30_day is None


def test_to_dict_returns_all_fields():
    assert __make().to_dict() == {
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
