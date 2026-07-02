from src.constants.filtering import NEGATIVE_TERMS, POSITIVE_TERMS, RULE_TOKEN_RE
from src.filtering.filter import Filter
from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode


class RuleBasedFilter(Filter):
    """A keyword-driven filter that scores each headline's directional sentiment.

    Every article is reduced to an integer signal: the number of bullish keywords
    minus the number of bearish keywords found in its title and description. The
    three modes then read that single signal differently:

    - BUY prefers the most positive signals (expected to rise).
    - SELL prefers the most negative signals (expected to fall).
    - AFFECT prefers the largest magnitude either way (expected to move most).

    The filter never looks at price data, so it makes a genuine prediction from
    text alone rather than peeking at the labels it is evaluated against.
    """

    def __init__(self, top_k: int = 10):
        """Create the filter with a selection size.

        Args:
            top_k (int): Number of articles to select from each batch.
        """
        self.__top_k = top_k

    @property
    def name(self) -> str:
        """The unique identifier of this filter."""
        return "rule_based"

    @property
    def top_k(self) -> int:
        """The number of articles the filter selects."""
        return self.__top_k

    def signal(self, article: SampledNews) -> int:
        """Return the bullish-minus-bearish keyword score for one article.

        Args:
            article (SampledNews): The article whose title and description are scored.

        Returns:
            int: Number of bullish keywords minus number of bearish keywords.
        """
        text = f"{article.news_title} {article.news_description or ''}".lower()
        tokens = RULE_TOKEN_RE.findall(text)
        positive = sum(1 for token in tokens if token in POSITIVE_TERMS)
        negative = sum(1 for token in tokens if token in NEGATIVE_TERMS)
        return positive - negative

    def select(
        self, articles: list[SampledNews], mode: FilterMode
    ) -> list[SampledNews]:
        """Rank the batch by keyword signal and return the top articles for the mode.

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
