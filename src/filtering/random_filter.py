import random

from src.filtering.filter import Filter
from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode


class RandomFilter(Filter):
    """A deliberately naive filter that ignores content and picks at random.

    It serves as the simplest possible strategy: regardless of mode it returns a
    seeded random subset of the batch, giving every other filter a floor to beat.
    """

    def __init__(self, top_k: int = 10, seed: int = 74):
        """Create the filter with a selection size and a seed for reproducibility.

        Args:
            top_k (int): Number of articles to select from each batch.
            seed (int): Seed for the internal random generator.
        """
        self.__top_k = top_k
        self.__seed = seed
        self.__rng = random.Random(seed)

    @property
    def name(self) -> str:
        """The unique identifier of this filter."""
        return "random"

    @property
    def top_k(self) -> int:
        """The number of articles the filter selects."""
        return self.__top_k

    @property
    def seed(self) -> int:
        """The seed of the internal random generator."""
        return self.__seed

    def select(
        self, articles: list[SampledNews], mode: FilterMode
    ) -> list[SampledNews]:
        """Return a seeded random subset of the batch, ignoring the mode.

        Args:
            articles (list[SampledNews]): The batch of candidate articles.
            mode (FilterMode): Ignored; random selection is mode-independent.

        Returns:
            list[SampledNews]: Up to top_k randomly chosen articles.
        """
        if len(articles) <= self.__top_k:
            return list(articles)
        return self.__rng.sample(articles, self.__top_k)
