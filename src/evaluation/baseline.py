from collections.abc import Sequence

from src.evaluation.news_scorer import NewsScorer
from src.models.baseline_reference import BaselineReference
from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode
from src.utils.enums.price_horizon import PriceHorizon


class Baseline:
    """Source of truth for a batch: the best, worst and average a filter could do.

    Given a batch of news and a selection size top_k, it answers, for every
    mode and horizon: what is the perfect selection worth (mean of the top_k
    best oriented values), what is the worst selection worth, and what does an
    average (random) pick yield. Filters are then judged against these anchors.
    """

    def __init__(self, articles: Sequence[SampledNews], top_k: int, scorer: NewsScorer | None = None):
        """Create the baseline for a batch and selection size.

        Args:
            articles (Sequence[SampledNews]): The batch to compute references over.
            top_k (int): Selection size the ideal and worst means are taken over.
            scorer (NewsScorer | None): Scorer for oriented values; a default one is used when None.
        """
        self.__articles = list(articles)
        self.__top_k = top_k
        self.__scorer = scorer if scorer is not None else NewsScorer()

    @property
    def top_k(self) -> int:
        """The selection size the ideal and worst means are taken over."""
        return self.__top_k

    def reference(self, mode: FilterMode, horizon: PriceHorizon) -> BaselineReference:
        """Build the ideal/average/worst reference for one mode and horizon.

        Args:
            mode (FilterMode): The selection mode to build the reference for.
            horizon (PriceHorizon): The price horizon to build the reference for.

        Returns:
            BaselineReference: The ideal, average and worst oriented values.
        """
        pairs = self.__scorer.oriented_pairs(self.__articles, mode, horizon)
        ordered = sorted(pairs, key=lambda pair: pair[1], reverse=True)
        values = [value for _, value in ordered]

        best = ordered[: self.__top_k]
        worst = ordered[-self.__top_k :] if ordered else []

        return BaselineReference(
            mode=mode,
            horizon=horizon,
            top_k=self.__top_k,
            ideal_value=NewsScorer.mean([value for _, value in best]),
            average_value=NewsScorer.mean(values),
            worst_value=NewsScorer.mean([value for _, value in worst]),
            ideal_selection=[article for article, _ in best],
        )

    def references(self) -> list[BaselineReference]:
        """Build references for every mode and horizon combination.

        Returns:
            list[BaselineReference]: One reference per mode and horizon pair.
        """
        return [
            self.reference(mode, horizon)
            for mode in FilterMode
            for horizon in PriceHorizon
        ]
