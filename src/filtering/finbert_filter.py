import functools

from transformers import pipeline

from src.filtering.filter import Filter
from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_NAME = "ProsusAI/finbert"
_INFERENCE_BATCH_SIZE = 32


@functools.lru_cache(maxsize=1)
def _load_pipeline():
    """Load the FinBERT classification pipeline once and share it everywhere.

    The model (~440 MB) is downloaded on first use and cached by the lru_cache so
    that constructing several filter instances (one per batch-size run) reuses a
    single in-memory model instead of reloading it each time.
    """
    logger.info(f"Loading FinBERT model '{_MODEL_NAME}' (downloads on first run).")
    return pipeline("text-classification", model=_MODEL_NAME, top_k=None, truncation=True)


class FinbertFilter(Filter):
    """Ranks headlines with FinBERT, a BERT model fine-tuned on financial text.

    FinBERT classifies each headline into ``positive`` / ``negative`` / ``neutral``
    with a probability for each. Those are collapsed into one directional signal:

        signal = P(positive) - P(negative)   in [-1, 1]

    which the three modes read like the other sentiment filters:

    - ``BUY`` prefers the most positive sentiment (expected to rise).
    - ``SELL`` prefers the most negative sentiment (expected to fall).
    - ``AFFECT`` prefers the largest magnitude either way (expected to move most).

    The evaluator asks the filter once per mode on the same batch, so the model
    would otherwise score every article three times. With ``is_cached=True`` the
    per-article signals are computed once and reused across modes, cutting CPU
    inference roughly threefold. The flag is kept switchable (and reflected in
    :attr:`name`) so the cached and uncached variants can be benchmarked side by
    side. The filter never looks at price data, so it predicts from text alone.
    """

    def __init__(self, top_k: int = 10, is_cached: bool = False):
        self.__top_k = top_k
        self.__is_cached = is_cached
        self.__pipeline = _load_pipeline()
        self.__cache: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "finbert_cached" if self.__is_cached else "finbert"

    @property
    def top_k(self) -> int:
        return self.__top_k

    @property
    def is_cached(self) -> bool:
        return self.__is_cached

    @staticmethod
    def __text(article: SampledNews) -> str:
        return f"{article.news_title} {article.news_description or ''}".strip()

    @staticmethod
    def __compound(scores: list[dict]) -> float:
        """Collapse FinBERT's per-label scores into P(positive) - P(negative)."""
        by_label = {entry["label"].lower(): entry["score"] for entry in scores}
        return by_label.get("positive", 0.0) - by_label.get("negative", 0.0)

    def __infer(self, articles: list[SampledNews]) -> list[float]:
        """Run FinBERT over the articles and return one signal per article."""
        if not articles:
            return []
        texts = [self.__text(article) for article in articles]
        results = self.__pipeline(texts, truncation=True, batch_size=_INFERENCE_BATCH_SIZE)
        return [self.__compound(scores) for scores in results]

    def __signals(self, articles: list[SampledNews]) -> list[float]:
        """Return signals for the articles, reusing cached values when enabled."""
        if not self.__is_cached:
            return self.__infer(articles)
        missing = [article for article in articles if article.news_id not in self.__cache]
        for article, value in zip(missing, self.__infer(missing)):
            self.__cache[article.news_id] = value
        return [self.__cache[article.news_id] for article in articles]

    def signal(self, article: SampledNews) -> float:
        """Return FinBERT's directional sentiment for one article in ``[-1, 1]``."""
        return self.__signals([article])[0]

    def select(
        self, articles: list[SampledNews], mode: FilterMode
    ) -> list[SampledNews]:
        if len(articles) <= self.__top_k:
            return list(articles)
        signals = self.__signals(list(articles))
        scored = list(zip(articles, signals))
        if mode is FilterMode.BUY:
            scored.sort(key=lambda pair: pair[1], reverse=True)
        elif mode is FilterMode.SELL:
            scored.sort(key=lambda pair: pair[1])
        else:
            scored.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return [article for article, _ in scored[: self.__top_k]]
