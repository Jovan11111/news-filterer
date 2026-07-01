from collections.abc import Callable, Sequence
from pathlib import Path

from src.evaluation.filter_evaluator import FilterEvaluator
from src.filtering.filter import Filter
from src.dataset.news_sampler import NewsSampler
from src.models.filter_run import FilterRun
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FilterRunner:
    """Runs a filter over a single dataset at several batch sizes.

    For each requested size it draws a fresh random sample (no per-ticker gap, so
    large batches are reachable), rebuilds the filter from the factory for
    reproducibility, and evaluates it. The result is one :class:`FilterRun` per
    size, ready for the report.
    """

    def __init__(self, dataset: Path, top_k: int, seed: int, subtract_market: bool = False, standardize: bool = False):
        self.__dataset = dataset
        self.__top_k = top_k
        self.__seed = seed
        self.__subtract_market = subtract_market
        self.__standardize = standardize

    @property
    def dataset(self) -> Path:
        return self.__dataset

    def run(self, filter_factory: Callable[[], Filter], batch_sizes: Sequence[int]) -> list[FilterRun]:
        """Evaluate the filter at every batch size and return the runs in order."""
        evaluator = FilterEvaluator(self.__top_k, self.__subtract_market, self.__standardize)
        runs: list[FilterRun] = []
        for size in batch_sizes:
            sampler = NewsSampler(database=self.__dataset, batch_size=size, min_gap_days=0, seed=self.__seed)
            batch = sampler.sample()
            if not batch:
                logger.warning(f"Batch size {size} produced no sample from {self.__dataset.name}; skipping.")
                continue
            evaluation = evaluator.evaluate(filter_factory(), batch)
            logger.info(f"Run size {size}: sampled {evaluation.batch_size} articles in {evaluation.elapsed_seconds * 1000:.1f} ms of filtering.")
            runs.append(FilterRun(requested_size=size, evaluation=evaluation))
        return runs
