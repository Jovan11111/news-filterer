from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode
from src.utils.enums.price_horizon import PriceHorizon


class BaselineReference:
    """Source-of-truth oriented values for one mode and horizon over a batch.

    All three values live in the mode's oriented space (higher is better):

    - ``ideal_value`` is the mean of the ``top_k`` best oriented values.
    - ``average_value`` is the mean over the whole batch (the random-pick floor).
    - ``worst_value`` is the mean of the ``top_k`` worst oriented values.
    """

    def __init__(
        self,
        mode: FilterMode,
        horizon: PriceHorizon,
        top_k: int,
        ideal_value: float | None,
        average_value: float | None,
        worst_value: float | None,
        ideal_selection: list[SampledNews],
    ):
        """Store the source-of-truth values for one mode and horizon over a batch.

        Args:
            mode (FilterMode): The selection mode the values were computed for.
            horizon (PriceHorizon): The price horizon the values were computed for.
            top_k (int): Selection size the ideal and worst means were taken over.
            ideal_value (float | None): Mean of the top_k best oriented values.
            average_value (float | None): Mean oriented value over the whole batch.
            worst_value (float | None): Mean of the top_k worst oriented values.
            ideal_selection (list[SampledNews]): The top_k articles forming the ideal pick.
        """
        self.__mode = mode
        self.__horizon = horizon
        self.__top_k = top_k
        self.__ideal_value = ideal_value
        self.__average_value = average_value
        self.__worst_value = worst_value
        self.__ideal_selection = ideal_selection

    @property
    def mode(self) -> FilterMode:
        """The selection mode the values were computed for."""
        return self.__mode

    @property
    def horizon(self) -> PriceHorizon:
        """The price horizon the values were computed for."""
        return self.__horizon

    @property
    def top_k(self) -> int:
        """The selection size the ideal and worst means were taken over."""
        return self.__top_k

    @property
    def ideal_value(self) -> float | None:
        """The mean of the top_k best oriented values (the perfect pick)."""
        return self.__ideal_value

    @property
    def average_value(self) -> float | None:
        """The mean oriented value over the whole batch (the random-pick floor)."""
        return self.__average_value

    @property
    def worst_value(self) -> float | None:
        """The mean of the top_k worst oriented values."""
        return self.__worst_value

    @property
    def ideal_selection(self) -> list[SampledNews]:
        """The top_k articles that form the ideal selection."""
        return self.__ideal_selection
