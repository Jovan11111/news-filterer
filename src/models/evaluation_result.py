from src.models.filter_score import FilterScore
from src.utils.enums.filter_mode import FilterMode
from src.utils.enums.price_horizon import PriceHorizon


class EvaluationResult:
    """A filter's full evaluation over one batch: all mode/horizon scores."""

    def __init__(
        self,
        filter_name: str,
        batch_size: int,
        top_k: int,
        scores: list[FilterScore],
        elapsed_seconds: float,
    ):
        self.__filter_name = filter_name
        self.__batch_size = batch_size
        self.__top_k = top_k
        self.__scores = scores
        self.__elapsed_seconds = elapsed_seconds

    @property
    def filter_name(self) -> str:
        return self.__filter_name

    @property
    def elapsed_seconds(self) -> float:
        return self.__elapsed_seconds

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @property
    def top_k(self) -> int:
        return self.__top_k

    @property
    def scores(self) -> list[FilterScore]:
        return self.__scores

    def score(self, mode: FilterMode, horizon: PriceHorizon) -> FilterScore | None:
        """Return the score for one mode/horizon pair, or ``None`` when missing."""
        for score in self.__scores:
            if score.mode is mode and score.horizon is horizon:
                return score
        return None
