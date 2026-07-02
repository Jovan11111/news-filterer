from src.models.evaluation_result import EvaluationResult


class FilterRun:
    """One filter evaluation at a requested batch size for the scalability report.

    requested_size is what we asked the sampler for; evaluation carries
    the actual sample size, the per-mode/horizon scores and the filter timing.
    """

    def __init__(self, requested_size: int, evaluation: EvaluationResult):
        """Store one filter evaluation at a requested batch size.

        Args:
            requested_size (int): Batch size the sampler was asked for.
            evaluation (EvaluationResult): The evaluation produced for that size.
        """
        self.__requested_size = requested_size
        self.__evaluation = evaluation

    @property
    def requested_size(self) -> int:
        """The batch size the sampler was asked for."""
        return self.__requested_size

    @property
    def evaluation(self) -> EvaluationResult:
        """The evaluation produced for the requested batch size."""
        return self.__evaluation

    def to_dict(self) -> dict:
        """Return this filter run as a dict."""
        return {
            "requested_size": self.__requested_size,
            "evaluation": self.__evaluation.to_dict(),
        }
