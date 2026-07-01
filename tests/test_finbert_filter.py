from datetime import date

from src.filtering.finbert_filter import FinbertFilter
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
    article = make_article(1, "Company reports record profit and raises guidance", "")
    assert FinbertFilter().signal(article) > 0


def test_signal_is_negative_for_bearish_text():
    article = make_article(1, "Company reports massive loss and cuts its dividend", "")
    assert FinbertFilter().signal(article) < 0


def test_signal_is_in_compound_range():
    article = make_article(1, "Shares climb after strong earnings", "")
    assert -1.0 <= FinbertFilter().signal(article) <= 1.0


def test_signal_tolerates_missing_description():
    article = make_article(1, "Earnings beat expectations", None)
    assert -1.0 <= FinbertFilter().signal(article) <= 1.0


def test_signal_is_deterministic():
    article = make_article(1, "Revenue grows on robust demand", "")
    filter_instance = FinbertFilter()
    assert filter_instance.signal(article) == filter_instance.signal(article)


def _batch() -> list[SampledNews]:
    return [
        make_article(1, "Record profit as revenue beats and guidance is raised", ""),  # positive
        make_article(2, "Massive loss as company cuts dividend amid fraud probe", ""),  # negative
        make_article(3, "Company to hold its annual shareholder meeting next month", ""),  # neutral
        make_article(4, "Shares gain after solid quarterly results", ""),  # positive
        make_article(5, "Outlook slashed on weak, disappointing demand", ""),  # negative
    ]


def test_buy_picks_a_positive_article_first():
    selected = FinbertFilter(top_k=1).select(_batch(), FilterMode.BUY)
    assert selected[0].id in {1, 4}


def test_sell_picks_a_negative_article_first():
    selected = FinbertFilter(top_k=1).select(_batch(), FilterMode.SELL)
    assert selected[0].id in {2, 5}


def test_affect_prefers_strong_sentiment_over_neutral():
    selected = FinbertFilter(top_k=2).select(_batch(), FilterMode.AFFECT)
    assert 3 not in [a.id for a in selected]


def test_select_returns_all_when_batch_not_larger_than_top_k():
    batch = _batch()
    selected = FinbertFilter(top_k=5).select(batch, FilterMode.BUY)
    assert selected == batch


def test_select_does_not_mutate_input():
    batch = _batch()
    original = list(batch)
    FinbertFilter(top_k=2).select(batch, FilterMode.BUY)
    assert batch == original


def test_select_is_deterministic():
    batch = _batch()
    first = FinbertFilter(top_k=3).select(batch, FilterMode.AFFECT)
    second = FinbertFilter(top_k=3).select(batch, FilterMode.AFFECT)
    assert [a.id for a in first] == [a.id for a in second]


def test_name_is_stable():
    assert FinbertFilter().name == "finbert"


def test_cached_name_differs():
    assert FinbertFilter(is_cached=True).name == "finbert_cached"
    assert FinbertFilter().is_cached is False
    assert FinbertFilter(is_cached=True).is_cached is True


def test_cached_signal_matches_uncached():
    article = make_article(1, "Record profit as revenue beats guidance", "")
    assert FinbertFilter(is_cached=True).signal(article) == FinbertFilter().signal(article)


def test_cached_select_matches_uncached():
    batch = _batch()
    uncached = FinbertFilter(top_k=2).select(batch, FilterMode.BUY)
    cached = FinbertFilter(top_k=2, is_cached=True).select(batch, FilterMode.BUY)
    assert [a.id for a in uncached] == [a.id for a in cached]


def test_cached_reuses_across_modes():
    batch = _batch()
    filter_instance = FinbertFilter(top_k=2, is_cached=True)
    first = filter_instance.select(batch, FilterMode.BUY)
    # A second mode call must reuse the cache and stay consistent with the first.
    second = filter_instance.select(batch, FilterMode.SELL)
    assert [a.id for a in first] != [] and [a.id for a in second] != []
    # Re-running BUY after SELL yields the identical selection (deterministic cache).
    assert [a.id for a in first] == [a.id for a in filter_instance.select(batch, FilterMode.BUY)]

