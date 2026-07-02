from abc import ABC, abstractmethod

from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode


class Filter(ABC):
    """Strategy interface shared by all news filters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """A short, unique identifier used in reports and figure file names."""

    @abstractmethod
    def select(
        self, articles: list[SampledNews], mode: FilterMode
    ) -> list[SampledNews]:
        """Return the subset of articles selected for the given mode.

        Implementations must not mutate articles and must return only
        elements drawn from the input batch.

        Args:
            articles (list[SampledNews]): The batch of candidate articles.
            mode (FilterMode): Whether to rank for BUY, SELL or AFFECT.

        Returns:
            list[SampledNews]: The selected (possibly empty) subset of the batch.
        """
