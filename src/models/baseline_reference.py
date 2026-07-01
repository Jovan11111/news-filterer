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
        self.__mode = mode
        self.__horizon = horizon
        self.__top_k = top_k
        self.__ideal_value = ideal_value
        self.__average_value = average_value
        self.__worst_value = worst_value
        self.__ideal_selection = ideal_selection

    @property
    def mode(self) -> FilterMode:
        return self.__mode

    @property
    def horizon(self) -> PriceHorizon:
        return self.__horizon

    @property
    def top_k(self) -> int:
        return self.__top_k

    @property
    def ideal_value(self) -> float | None:
        return self.__ideal_value

    @property
    def average_value(self) -> float | None:
        return self.__average_value

    @property
    def worst_value(self) -> float | None:
        return self.__worst_value

    @property
    def ideal_selection(self) -> list[SampledNews]:
        return self.__ideal_selection
