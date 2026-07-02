from datetime import date

from src.models.news_item import NewsItem


def __make() -> NewsItem:
    return NewsItem(
        news_id="n1",
        title="Headline",
        description="Body",
        ticker="AAPL",
        pub_date=date(2026, 1, 2),
    )


def test_properties_return_stored_values():
    item = __make()
    assert item.news_id == "n1"
    assert item.title == "Headline"
    assert item.description == "Body"
    assert item.ticker == "AAPL"
    assert item.pub_date == date(2026, 1, 2)


def test_to_dict_returns_all_fields():
    assert __make().to_dict() == {
        "news_id": "n1",
        "title": "Headline",
        "description": "Body",
        "ticker": "AAPL",
        "pub_date": "2026-01-02",
    }
