from collections.abc import Sequence
from math import sqrt

from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode
from src.utils.enums.price_horizon import PriceHorizon


class NewsScorer:
    """Turns a news article's price change into a mode-oriented value.

    The three filter modes are unified into a single "higher is better" value so
    that baselines and evaluation share one comparison axis:

    - ``BUY`` keeps the signed change (a bigger gain is better).
    - ``SELL`` negates the change (a bigger drop is better).
    - ``AFFECT`` takes the absolute change (a bigger move either way is better).

    When ``subtract_market`` is set the broad-market (S&P 500) change over the
    same horizon is subtracted first, so the value reflects the news-driven
    (abnormal) move rather than the whole market's drift. Articles missing the
    market change fall back to their raw price change.

    When ``standardize`` is set the (market-adjusted) move is divided by the
    stock's own trailing volatility scaled to the horizon, expressing it in
    standard deviations (a z-score). This normalizes for the fact that some
    stocks routinely move more than others. Articles without a usable
    volatility estimate are dropped, since their z-score is undefined.
    """

    # Approximate number of trading days spanned by each horizon, used to scale
    # the daily volatility up to the horizon when standardizing a price move into
    # a z-score (a 7-day window covers roughly five trading sessions).
    __HORIZON_TRADING_DAYS = {PriceHorizon.ONE_DAY: 1, PriceHorizon.SEVEN_DAYS: 5}

    def __init__(self, subtract_market: bool = False, standardize: bool = False):
        """Create the scorer with the label transforms to apply.

        Args:
            subtract_market (bool): Whether to subtract the S&P 500 change (abnormal return).
            standardize (bool): Whether to divide the move by 30-day volatility (z-score).
        """
        self.__subtract_market = subtract_market
        self.__standardize = standardize

    @property
    def subtract_market(self) -> bool:
        """Whether the S&P 500 change is subtracted (abnormal return)."""
        return self.__subtract_market

    @property
    def standardize(self) -> bool:
        """Whether the move is divided by 30-day volatility (z-score)."""
        return self.__standardize

    def price_change(self, article: SampledNews, horizon: PriceHorizon) -> float | None:
        """Return the price move for one horizon, market-adjusted and/or standardized when enabled.

        Args:
            article (SampledNews): The article to read the price and market change from.
            horizon (PriceHorizon): The horizon (1-day or 7-day) to read.

        Returns:
            float | None: The (possibly adjusted) move, or None when required data is missing.
        """
        if horizon is PriceHorizon.ONE_DAY:
            price, market = article.price_change_1_day, article.market_change_1_day
        else:
            price, market = article.price_change_7_days, article.market_change_7_days
        if price is None:
            return None
        value = price
        if self.__subtract_market and market is not None:
            value = price - market
        if self.__standardize:
            volatility = article.volatility_30_day
            if not volatility:
                return None
            value /= volatility * sqrt(self.__HORIZON_TRADING_DAYS[horizon])
        return value

    @staticmethod
    def oriented_value(change: float, mode: FilterMode) -> float:
        """Map a signed price change to a mode-oriented "higher is better" value.

        Args:
            change (float): The signed price change.
            mode (FilterMode): BUY keeps the sign, SELL negates it, AFFECT takes the magnitude.

        Returns:
            float: The oriented value where higher is always better.
        """
        if mode is FilterMode.BUY:
            return change
        if mode is FilterMode.SELL:
            return -change
        return abs(change)

    def oriented_pairs(self, articles: Sequence[SampledNews], mode: FilterMode, horizon: PriceHorizon) -> list[tuple[SampledNews, float]]:
        """Pair each article with its oriented value, dropping those with no change for the horizon.

        Args:
            articles (Sequence[SampledNews]): The articles to score.
            mode (FilterMode): The selection mode that orients the values.
            horizon (PriceHorizon): The price horizon to score on.

        Returns:
            list[tuple[SampledNews, float]]: Article/oriented-value pairs for scorable articles.
        """
        pairs: list[tuple[SampledNews, float]] = []
        for article in articles:
            change = self.price_change(article, horizon)
            if change is None:
                continue
            pairs.append((article, self.oriented_value(change, mode)))
        return pairs

    def oriented_values(self, articles: Sequence[SampledNews], mode: FilterMode, horizon: PriceHorizon) -> list[float]:
        """Return only the oriented values for the given articles.

        Args:
            articles (Sequence[SampledNews]): The articles to score.
            mode (FilterMode): The selection mode that orients the values.
            horizon (PriceHorizon): The price horizon to score on.

        Returns:
            list[float]: The oriented values of the scorable articles.
        """
        return [value for _, value in self.oriented_pairs(articles, mode, horizon)]

    def mean_oriented_value(self, articles: Sequence[SampledNews], mode: FilterMode, horizon: PriceHorizon) -> float | None:
        """Return the mean oriented value over the articles, or None when none have data.

        Args:
            articles (Sequence[SampledNews]): The articles to score.
            mode (FilterMode): The selection mode that orients the values.
            horizon (PriceHorizon): The price horizon to score on.

        Returns:
            float | None: The mean oriented value, or None when no article is scorable.
        """
        return self.mean(self.oriented_values(articles, mode, horizon))

    @staticmethod
    def mean(values: Sequence[float]) -> float | None:
        """Return the arithmetic mean of a sequence, or None when it is empty.

        Args:
            values (Sequence[float]): The values to average.

        Returns:
            float | None: The mean, or None when the sequence is empty.
        """
        return sum(values) / len(values) if values else None
