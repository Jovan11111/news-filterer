from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.filtering.filter import Filter
from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode


class VaderSentimentFilter(Filter):
    """Ranks headlines by VADER's compound sentiment score.

    VADER is a lexicon-and-rules sentiment model that returns a ``compound``
    score in ``[-1, +1]`` for a piece of text, accounting for intensifiers,
    negations, punctuation and capitalisation. It is the natural step up from the
    keyword-counting rule-based filter while still needing no API key, no model
    download and staying fully deterministic.

    The single compound signal is read per mode, exactly like the rule-based
    filter:

    - ``BUY`` prefers the most positive sentiment (expected to rise).
    - ``SELL`` prefers the most negative sentiment (expected to fall).
    - ``AFFECT`` prefers the largest magnitude either way (expected to move most).

    The filter never looks at price data, so it predicts from text alone.
    """

    def __init__(self, top_k: int = 10):
        """Create the filter with a selection size and a VADER analyzer.

        Args:
            top_k (int): Number of articles to select from each batch.
        """
        self.__top_k = top_k
        self.__analyzer = SentimentIntensityAnalyzer()

    @property
    def name(self) -> str:
        """The unique identifier of this filter."""
        return "vader"

    @property
    def top_k(self) -> int:
        """The number of articles the filter selects."""
        return self.__top_k

    def signal(self, article: SampledNews) -> float:
        """Return VADER's compound sentiment score for one article in ``[-1, 1]``.

        Args:
            article (SampledNews): The article whose title and description are scored.

        Returns:
            float: The VADER compound sentiment score, from -1 (negative) to 1 (positive).
        """
        text = f"{article.news_title} {article.news_description or ''}".strip()
        return self.__analyzer.polarity_scores(text)["compound"]

    def select(
        self, articles: list[SampledNews], mode: FilterMode
    ) -> list[SampledNews]:
        """Rank the batch by VADER sentiment and return the top articles for the mode.

        Args:
            articles (list[SampledNews]): The batch of candidate articles.
            mode (FilterMode): Whether to rank for BUY, SELL or AFFECT.

        Returns:
            list[SampledNews]: Up to top_k articles ordered best-first for the mode.
        """
        if len(articles) <= self.__top_k:
            return list(articles)
        scored = [(article, self.signal(article)) for article in articles]
        if mode is FilterMode.BUY:
            scored.sort(key=lambda pair: pair[1], reverse=True)
        elif mode is FilterMode.SELL:
            scored.sort(key=lambda pair: pair[1])
        else:
            scored.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return [article for article, _ in scored[: self.__top_k]]
