class ExperimentConfig:
    """Settings that drive the filtering and report scripts, read from config/experiment_config.json."""

    def __init__(self, filter_name: str, batch_size: int, top_k: int, seed: int, scalability_sizes: list[int], subtract_market: bool, standardize: bool):
        """Store the settings that drive the filtering and report scripts.

        Args:
            filter_name (str): Name of the filter the report runs.
            batch_size (int): Sample size used by the single-batch filter script.
            top_k (int): Number of articles each filter selects.
            seed (int): Random seed for reproducible sampling.
            scalability_sizes (list[int]): Batch sizes the report sweeps over.
            subtract_market (bool): Whether to use market-adjusted (abnormal) returns.
            standardize (bool): Whether to divide moves by 30-day volatility (z-scores).
        """
        self.__filter_name = filter_name
        self.__batch_size = batch_size
        self.__top_k = top_k
        self.__seed = seed
        self.__scalability_sizes = scalability_sizes
        self.__subtract_market = subtract_market
        self.__standardize = standardize

    @property
    def filter_name(self) -> str:
        """The name of the filter the report runs."""
        return self.__filter_name

    @property
    def batch_size(self) -> int:
        """The sample size used by the single-batch filter script."""
        return self.__batch_size

    @property
    def top_k(self) -> int:
        """The number of articles each filter selects."""
        return self.__top_k

    @property
    def seed(self) -> int:
        """The random seed for reproducible sampling."""
        return self.__seed

    @property
    def scalability_sizes(self) -> list[int]:
        """The batch sizes the report sweeps over."""
        return self.__scalability_sizes

    @property
    def subtract_market(self) -> bool:
        """Whether to use market-adjusted (abnormal) returns."""
        return self.__subtract_market

    @property
    def standardize(self) -> bool:
        """Whether to divide moves by 30-day volatility (z-scores)."""
        return self.__standardize

    def to_dict(self) -> dict:
        """Return this experiment config as a dict."""
        return {
            "filter_name": self.__filter_name,
            "batch_size": self.__batch_size,
            "top_k": self.__top_k,
            "seed": self.__seed,
            "scalability_sizes": list(self.__scalability_sizes),
            "subtract_market": self.__subtract_market,
            "standardize": self.__standardize,
        }
