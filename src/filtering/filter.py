from abc import ABC, abstractmethod

from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode


class Filter(ABC):
    """Strategy interface shared by all news filters."""

    @abstractmethod
    def select(
        self, articles: list[SampledNews], mode: FilterMode
    ) -> list[SampledNews]:
        """Return the subset of articles selected for mode.

        Implementations must not mutate articles and should return a (possibly
        empty) list containing only elements drawn from the input batch.
        """
