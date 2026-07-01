from datetime import date

from src.filtering.vader_sentiment_filter import VaderSentimentFilter
from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode


def make_article(article_id: int, title: str, description: str | None = "") -> SampledNews:
    """Build a SampledNews with only the text fields that the filter reads."""
    return SampledNews(
        id=article_id,
        news_id=f"n{article_id}",
        news_title=title,
        news_description=description,
        ticker="TEST",
        news_date=date(2025, 1, 1),
        price_change_1_day=None,
        price_change_7_days=None,
    )


def test_signal_is_positive_for_bullish_text():
    article = make_article(1, "Profit surges as revenue beats expectations", "record growth")
    assert VaderSentimentFilter().signal(article) > 0


def test_signal_is_negative_for_bearish_text():
    article = make_article(1, "Shares plunge after disastrous loss and fraud probe", "")
    assert VaderSentimentFilter().signal(article) < 0


def test_signal_is_in_compound_range():
    article = make_article(1, "Stock rises on strong demand", "")
    assert -1.0 <= VaderSentimentFilter().signal(article) <= 1.0


def test_signal_tolerates_missing_description():
    article = make_article(1, "Great earnings beat", None)
    assert VaderSentimentFilter().signal(article) > 0


def test_signal_is_deterministic():
    article = make_article(1, "Company posts excellent results", "")
    filter_instance = VaderSentimentFilter()
    assert filter_instance.signal(article) == filter_instance.signal(article)


def _batch() -> list[SampledNews]:
    return [
        make_article(1, "Outstanding profit, revenue beats records", "strong growth and success"),  # very positive
        make_article(2, "Catastrophic loss as fraud probe and lawsuit hit shares", ""),  # very negative
        make_article(3, "Company schedules its annual meeting", "routine update"),  # neutral
        make_article(4, "Stock gains on good news", ""),  # mildly positive
        make_article(5, "Guidance cut amid weak, disappointing demand", ""),  # negative
    ]


def test_buy_picks_most_positive_first():
    selected = VaderSentimentFilter(top_k=1).select(_batch(), FilterMode.BUY)
    assert selected[0].id == 1


def test_sell_picks_most_negative_first():
    selected = VaderSentimentFilter(top_k=1).select(_batch(), FilterMode.SELL)
    assert selected[0].id == 2


def test_affect_prefers_strong_sentiment_over_neutral():
    selected = VaderSentimentFilter(top_k=2).select(_batch(), FilterMode.AFFECT)
    # The neutral article (3) must not be among the two most affecting.
    assert 3 not in [a.id for a in selected]


def test_select_returns_all_when_batch_not_larger_than_top_k():
    batch = _batch()
    selected = VaderSentimentFilter(top_k=5).select(batch, FilterMode.BUY)
    assert selected == batch


def test_select_does_not_mutate_input():
    batch = _batch()
    original = list(batch)
    VaderSentimentFilter(top_k=2).select(batch, FilterMode.BUY)
    assert batch == original


def test_select_is_deterministic():
    batch = _batch()
    first = VaderSentimentFilter(top_k=3).select(batch, FilterMode.AFFECT)
    second = VaderSentimentFilter(top_k=3).select(batch, FilterMode.AFFECT)
    assert [a.id for a in first] == [a.id for a in second]


def test_name_is_stable():
    assert VaderSentimentFilter().name == "vader"
