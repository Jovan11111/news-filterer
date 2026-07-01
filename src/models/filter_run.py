from src.models.evaluation_result import EvaluationResult


class FilterRun:
    """One filter evaluation at a requested batch size for the scalability report.

    ``requested_size`` is what we asked the sampler for; ``evaluation`` carries
    the actual sample size, the per-mode/horizon scores and the filter timing.
    """

    def __init__(self, requested_size: int, evaluation: EvaluationResult):
        self.__requested_size = requested_size
        self.__evaluation = evaluation

    @property
    def requested_size(self) -> int:
        return self.__requested_size

    @property
    def evaluation(self) -> EvaluationResult:
        return self.__evaluation
