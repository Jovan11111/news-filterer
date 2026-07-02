from src.utils.enums.filter_mode import FilterMode
from src.utils.enums.price_horizon import PriceHorizon


class FilterScore:
    """How a filter's selection compares to the baseline for one mode/horizon.

    Values are in the mode's oriented space (higher is better). gap_to_ideal
    is how much the filter falls short of the perfect pick, lift_over_average
    is how far it beats the random floor, and normalized_score places the
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
        """Store how a filter's selection compares to the baseline for one mode/horizon.

        Args:
            mode (FilterMode): The selection mode the score belongs to.
            horizon (PriceHorizon): The price horizon the score belongs to.
            top_k (int): Selection size the values were computed over.
            filter_value (float | None): Mean oriented value of the filter's picks.
            ideal_value (float | None): Mean oriented value of the perfect pick.
            average_value (float | None): Mean oriented value of a random pick.
            worst_value (float | None): Mean oriented value of the worst pick.
            gap_to_ideal (float | None): How far the filter falls short of the ideal.
            lift_over_average (float | None): How far the filter beats the random floor.
            normalized_score (float | None): Score on a 0..1 scale (0 = random, 1 = ideal).
        """
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
        """The selection mode the score belongs to."""
        return self.__mode

    @property
    def horizon(self) -> PriceHorizon:
        """The price horizon the score belongs to."""
        return self.__horizon

    @property
    def top_k(self) -> int:
        """The selection size the values were computed over."""
        return self.__top_k

    @property
    def filter_value(self) -> float | None:
        """The mean oriented value of the filter's selection."""
        return self.__filter_value

    @property
    def ideal_value(self) -> float | None:
        """The mean oriented value of the perfect selection."""
        return self.__ideal_value

    @property
    def average_value(self) -> float | None:
        """The mean oriented value of a random selection (the floor)."""
        return self.__average_value

    @property
    def worst_value(self) -> float | None:
        """The mean oriented value of the worst selection."""
        return self.__worst_value

    @property
    def gap_to_ideal(self) -> float | None:
        """How far the filter falls short of the ideal selection."""
        return self.__gap_to_ideal

    @property
    def lift_over_average(self) -> float | None:
        """How far the filter beats the random-pick floor."""
        return self.__lift_over_average

    @property
    def normalized_score(self) -> float | None:
        """The score on a 0..1 scale (0 = no better than random, 1 = ideal)."""
        return self.__normalized_score

    def to_dict(self) -> dict:
        """Return this filter score as a dict."""
        return {
            "mode": self.__mode.value,
            "horizon": self.__horizon.value,
            "top_k": self.__top_k,
            "filter_value": self.__filter_value,
            "ideal_value": self.__ideal_value,
            "average_value": self.__average_value,
            "worst_value": self.__worst_value,
            "gap_to_ideal": self.__gap_to_ideal,
            "lift_over_average": self.__lift_over_average,
            "normalized_score": self.__normalized_score,
        }
