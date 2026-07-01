from src.utils.enums.filter_mode import FilterMode
from src.utils.enums.price_horizon import PriceHorizon


class FilterScore:
    """How a filter's selection compares to the baseline for one mode/horizon.

    Values are in the mode's oriented space (higher is better). ``gap_to_ideal``
    is how much the filter falls short of the perfect pick, ``lift_over_average``
    is how far it beats the random floor, and ``normalized_score`` places the
    filter on a 0..1 scale where 0 means "no better than random" and 1 means
    "as good as the ideal selection".
    """

    def __init__(
        self,
        mode: FilterMode,
        horizon: PriceHorizon,
        top_k: int,
        filter_value: float | None,
        ideal_value: float | None,
        average_value: float | None,
        worst_value: float | None,
        gap_to_ideal: float | None,
        lift_over_average: float | None,
        normalized_score: float | None,
    ):
        self.__mode = mode
        self.__horizon = horizon
        self.__top_k = top_k
        self.__filter_value = filter_value
        self.__ideal_value = ideal_value
        self.__average_value = average_value
        self.__worst_value = worst_value
        self.__gap_to_ideal = gap_to_ideal
        self.__lift_over_average = lift_over_average
        self.__normalized_score = normalized_score

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
    def filter_value(self) -> float | None:
        return self.__filter_value

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
    def gap_to_ideal(self) -> float | None:
        return self.__gap_to_ideal

    @property
    def lift_over_average(self) -> float | None:
        return self.__lift_over_average

    @property
    def normalized_score(self) -> float | None:
        return self.__normalized_score
