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
        """Store a filter's full evaluation over one batch.

        Args:
            filter_name (str): Identifier of the evaluated filter.
            batch_size (int): Number of articles actually in the evaluated batch.
            top_k (int): Selection size the filter was asked for.
            scores (list[FilterScore]): One score per mode and horizon.
            elapsed_seconds (float): Total time the filter spent selecting, in seconds.
        """
        self.__filter_name = filter_name
        self.__batch_size = batch_size
        self.__top_k = top_k
        self.__scores = scores
        self.__elapsed_seconds = elapsed_seconds

    @property
    def filter_name(self) -> str:
        """The identifier of the evaluated filter."""
        return self.__filter_name

    @property
    def elapsed_seconds(self) -> float:
        """The total time the filter spent selecting, in seconds."""
        return self.__elapsed_seconds

    @property
    def batch_size(self) -> int:
        """The number of articles actually in the evaluated batch."""
        return self.__batch_size

    @property
    def top_k(self) -> int:
        """The selection size the filter was asked for."""
        return self.__top_k

    @property
    def scores(self) -> list[FilterScore]:
        """The list of scores, one per mode and horizon."""
        return self.__scores

    def score(self, mode: FilterMode, horizon: PriceHorizon) -> FilterScore | None:
        """Return the score for one mode and horizon pair.

        Args:
            mode (FilterMode): The selection mode to look up.
            horizon (PriceHorizon): The price horizon to look up.

        Returns:
            FilterScore | None: The matching score, or None when there is none.
        """
        for score in self.__scores:
            if score.mode is mode and score.horizon is horizon:
                return score
        return None

    def to_dict(self) -> dict:
        """Return this evaluation result as a dict."""
        return {
            "filter_name": self.__filter_name,
            "batch_size": self.__batch_size,
            "top_k": self.__top_k,
            "scores": [score.to_dict() for score in self.__scores],
            "elapsed_seconds": self.__elapsed_seconds,
        }
