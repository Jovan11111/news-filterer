from transformers import pipeline

from src.filtering.filter import Filter
from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode
from src.utils.logger import get_logger

logger = get_logger(__name__)


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

    __MODEL_NAME = "ProsusAI/finbert"
    __INFERENCE_BATCH_SIZE = 32

    # The FinBERT pipeline (~440 MB) is loaded once and shared across every
    # instance (one per batch-size run) instead of being reloaded each time.
    __shared_pipeline = None

    def __init__(self, top_k: int = 10, is_cached: bool = False):
        """Create the filter and obtain the shared FinBERT pipeline.

        Args:
            top_k (int): Number of articles to select from each batch.
            is_cached (bool): Whether to cache per-article signals across modes.
        """
        self.__top_k = top_k
        self.__is_cached = is_cached
        self.__pipeline = self.__load_pipeline()
        self.__cache: dict[str, float] = {}

    @classmethod
    def __load_pipeline(cls):
        """Load and cache the shared FinBERT pipeline, downloading it on first use."""
        if cls.__shared_pipeline is None:
            logger.info(f"Loading FinBERT model '{cls.__MODEL_NAME}' (downloads on first run).")
            cls.__shared_pipeline = pipeline("text-classification", model=cls.__MODEL_NAME, top_k=None, truncation=True)
        return cls.__shared_pipeline

    @property
    def name(self) -> str:
        """The unique identifier of this filter (cached or uncached variant)."""
        return "finbert_cached" if self.__is_cached else "finbert"

    @property
    def top_k(self) -> int:
        """The number of articles the filter selects."""
        return self.__top_k

    @property
    def is_cached(self) -> bool:
        """Whether per-article signals are cached across modes."""
        return self.__is_cached

    @staticmethod
    def __text(article: SampledNews) -> str:
        """Return the article's title and description joined into one string."""
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
        results = self.__pipeline(texts, truncation=True, batch_size=self.__INFERENCE_BATCH_SIZE)
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
        """Return FinBERT's directional sentiment for one article in ``[-1, 1]``.

        Args:
            article (SampledNews): The article whose title and description are scored.

        Returns:
            float: P(positive) - P(negative), from -1 (negative) to 1 (positive).
        """
        return self.__signals([article])[0]

    def select(
        self, articles: list[SampledNews], mode: FilterMode
    ) -> list[SampledNews]:
        """Rank the batch by FinBERT sentiment and return the top articles for the mode.

        Args:
            articles (list[SampledNews]): The batch of candidate articles.
            mode (FilterMode): Whether to rank for BUY, SELL or AFFECT.

        Returns:
            list[SampledNews]: Up to top_k articles ordered best-first for the mode.
        """
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
