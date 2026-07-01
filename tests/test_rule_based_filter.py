from datetime import date

from src.filtering.rule_based_filter import RuleBasedFilter
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


def test_signal_counts_positive_minus_negative():
    article = make_article(1, "Profit surges as revenue beats", "record growth")
    # surges, beats, record, growth, profit -> 5 positive, 0 negative
    assert RuleBasedFilter().signal(article) == 5


def test_signal_handles_negative_terms():
    article = make_article(1, "Shares plunge on profit warning", "lawsuit and layoffs")
    # plunge, warning, lawsuit, layoffs -> 4 negative; profit -> 1 positive => -3
    assert RuleBasedFilter().signal(article) == -3


def test_signal_is_zero_for_neutral_text():
    article = make_article(1, "Company schedules annual meeting", "details to follow")
    assert RuleBasedFilter().signal(article) == 0


def test_signal_is_case_insensitive():
    lower = make_article(1, "stock surges", "")
    upper = make_article(2, "STOCK SURGES", "")
    assert RuleBasedFilter().signal(lower) == RuleBasedFilter().signal(upper) == 1


def test_signal_tolerates_missing_description():
    article = make_article(1, "Earnings beat expectations", None)
    assert RuleBasedFilter().signal(article) == 1


def test_signal_ignores_substrings():
    # "winter" contains "win" but must not be counted as a keyword.
    article = make_article(1, "Winter weather slows shipments", "")
    assert RuleBasedFilter().signal(article) == 0


def _batch() -> list[SampledNews]:
    return [
        make_article(1, "Profit surges, revenue beats record", "strong growth"),  # +5
        make_article(2, "Shares plunge on fraud probe", "lawsuit warning"),  # -4
        make_article(3, "Company holds annual meeting", "routine update"),  # 0
        make_article(4, "Stock rises modestly", ""),  # +1
        make_article(5, "Guidance cut sparks selloff", "weak demand"),  # -3
    ]


def test_buy_picks_most_positive_first():
    selected = RuleBasedFilter(top_k=2).select(_batch(), FilterMode.BUY)
    assert [a.id for a in selected] == [1, 4]


def test_sell_picks_most_negative_first():
    selected = RuleBasedFilter(top_k=2).select(_batch(), FilterMode.SELL)
    assert [a.id for a in selected] == [2, 5]


def test_affect_picks_largest_magnitude():
    selected = RuleBasedFilter(top_k=2).select(_batch(), FilterMode.AFFECT)
    assert [a.id for a in selected] == [1, 2]


def test_select_returns_all_when_batch_not_larger_than_top_k():
    batch = _batch()
    selected = RuleBasedFilter(top_k=5).select(batch, FilterMode.BUY)
    assert selected == batch


def test_select_does_not_mutate_input():
    batch = _batch()
    original = list(batch)
    RuleBasedFilter(top_k=2).select(batch, FilterMode.BUY)
    assert batch == original


def test_select_is_deterministic():
    batch = _batch()
    first = RuleBasedFilter(top_k=3).select(batch, FilterMode.AFFECT)
    second = RuleBasedFilter(top_k=3).select(batch, FilterMode.AFFECT)
    assert [a.id for a in first] == [a.id for a in second]


def test_ties_keep_original_order():
    batch = [
        make_article(1, "neutral one", ""),
        make_article(2, "neutral two", ""),
        make_article(3, "neutral three", ""),
    ]
    selected = RuleBasedFilter(top_k=2).select(batch, FilterMode.BUY)
    assert [a.id for a in selected] == [1, 2]


def test_name_is_stable():
    assert RuleBasedFilter().name == "rule_based"
