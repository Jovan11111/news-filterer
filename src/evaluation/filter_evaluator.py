import time
from collections.abc import Sequence

from src.evaluation.baseline import Baseline
from src.evaluation.news_scorer import NewsScorer
from src.filtering.filter import Filter
from src.models.baseline_reference import BaselineReference
from src.models.evaluation_result import EvaluationResult
from src.models.filter_score import FilterScore
from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode
from src.utils.enums.price_horizon import PriceHorizon
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FilterEvaluator:
    """Runs a filter over a batch and scores its picks against the baseline.

    For every mode the filter is asked once for its selection; that selection is
    then scored on both price horizons against the batch's ideal/average/worst
    references, yielding one :class:`FilterScore` per mode and horizon.
    """

    def __init__(self, top_k: int, subtract_market: bool = False, standardize: bool = False):
        """Create the evaluator with a selection size and label options.

        Args:
            top_k (int): Number of articles each filter selects.
            subtract_market (bool): Whether to use market-adjusted (abnormal) returns.
            standardize (bool): Whether to divide moves by 30-day volatility (z-scores).
        """
        self.__top_k = top_k
        self.__scorer = NewsScorer(subtract_market, standardize)

    @property
    def top_k(self) -> int:
        """The number of articles each filter selects."""
        return self.__top_k

    @property
    def subtract_market(self) -> bool:
        """Whether market-adjusted (abnormal) returns are used."""
        return self.__scorer.subtract_market

    @property
    def standardize(self) -> bool:
        """Whether moves are divided by 30-day volatility (z-scores)."""
        return self.__scorer.standardize

    def evaluate(self, news_filter: Filter, articles: Sequence[SampledNews]) -> EvaluationResult:
        """Evaluate ``news_filter`` on ``articles`` across all modes and horizons.

        Args:
            news_filter (Filter): The filter to run and score.
            articles (Sequence[SampledNews]): The batch to evaluate the filter on.

        Returns:
            EvaluationResult: The filter's scores and timing over the batch.
        """
        batch = list(articles)
        baseline = Baseline(batch, self.__top_k, self.__scorer)

        scores: list[FilterScore] = []
        elapsed_seconds = 0.0
        for mode in FilterMode:
            start = time.perf_counter()
            selection = news_filter.select(batch, mode)
            elapsed_seconds += time.perf_counter() - start
            logger.info(f"{news_filter.name} selected {len(selection)} of {len(batch)} articles for mode '{mode.value}'.")
            for horizon in PriceHorizon:
                reference = baseline.reference(mode, horizon)
                filter_value = self.__scorer.mean_oriented_value(selection, mode, horizon)
                scores.append(self.__score(mode, horizon, filter_value, reference))

        return EvaluationResult(
            filter_name=news_filter.name,
            batch_size=len(batch),
            top_k=self.__top_k,
            scores=scores,
            elapsed_seconds=elapsed_seconds,
        )

    def __score(self, mode: FilterMode, horizon: PriceHorizon, filter_value: float | None, reference: BaselineReference) -> FilterScore:
        """Build one FilterScore comparing a filter value to its baseline reference."""
        ideal_value = reference.ideal_value
        average_value = reference.average_value

        gap_to_ideal = None
        lift_over_average = None
        normalized_score = None
        if filter_value is not None and ideal_value is not None:
            gap_to_ideal = ideal_value - filter_value
        if filter_value is not None and average_value is not None:
            lift_over_average = filter_value - average_value
        if filter_value is not None and ideal_value is not None and average_value is not None:
            spread = ideal_value - average_value
            normalized_score = (filter_value - average_value) / spread if spread != 0 else None

        return FilterScore(
            mode=mode,
            horizon=horizon,
            top_k=self.__top_k,
            filter_value=filter_value,
            ideal_value=ideal_value,
            average_value=average_value,
            worst_value=reference.worst_value,
            gap_to_ideal=gap_to_ideal,
            lift_over_average=lift_over_average,
            normalized_score=normalized_score,
        )
