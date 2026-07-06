from datetime import date

from src.filtering.trained_model_filter import TrainedModelFilter
from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode


def make_article(article_id: int, title: str, description: str | None = "") -> SampledNews:
    """Build a SampledNews with only the text fields the filter reads."""
    return SampledNews(
        id=article_id,
        news_id=f"n{article_id}",
        news_title=title,
        news_description=description,
        ticker="TEST",
        news_date=date(2026, 1, 1),
        price_change_1_day=None,
        price_change_7_days=None,
    )


class StubModel:
    """A fake predictor returning a preset predicted move per headline text."""

    def __init__(self, scores_by_text: dict[str, float]):
        self.__scores = scores_by_text

    def predict(self, texts):
        return [self.__scores[text] for text in texts]


_SCORES = {"Rise": 3.0, "Fall": -5.0, "Flat": 0.5, "Small": -0.2, "Mid": 1.0}


def _filter(top_k: int = 2) -> TrainedModelFilter:
    return TrainedModelFilter(top_k=top_k, model=StubModel(_SCORES))


def _batch() -> list[SampledNews]:
    return [make_article(i, title) for i, title in enumerate(_SCORES, start=1)]


def test_name_and_top_k():
    news_filter = _filter(top_k=5)
    assert news_filter.name == "trained_model"
    assert news_filter.top_k == 5


def test_signal_returns_model_prediction():
    news_filter = _filter()
    assert news_filter.signal(make_article(1, "Rise")) == 3.0
    assert news_filter.signal(make_article(2, "Fall")) == -5.0


def test_buy_prefers_the_largest_predicted_rise():
    picks = _filter().select(_batch(), FilterMode.BUY)
    assert [article.news_title for article in picks] == ["Rise", "Mid"]


def test_sell_prefers_the_largest_predicted_fall():
    picks = _filter().select(_batch(), FilterMode.SELL)
    assert [article.news_title for article in picks] == ["Fall", "Small"]


def test_affect_prefers_the_largest_magnitude():
    picks = _filter().select(_batch(), FilterMode.AFFECT)
    assert [article.news_title for article in picks] == ["Fall", "Rise"]


def test_small_batch_returns_all_articles():
    batch = _batch()[:2]
    picks = _filter(top_k=5).select(batch, FilterMode.BUY)
    assert picks == batch


def test_select_does_not_mutate_input():
    batch = _batch()
    original = list(batch)
    _filter().select(batch, FilterMode.BUY)
    assert batch == original
