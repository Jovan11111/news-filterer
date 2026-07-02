import json
from datetime import date
from pathlib import Path

from src.models.dataset_data import DatasetData
from src.models.experiment_config import ExperimentConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class JsonDecoder:
    """Class used for decoding a .json config file."""

    @staticmethod
    def decode(path_to_json_file: Path) -> DatasetData:
        """Decode the config .json file to a dataset data model.

        Args:
            path_to_json_file (Path): Path to config file.

        Returns:
            DatasetData: Model that contains the tickers list name and date range.
        """
        logger.debug("Decoding .json config file.")
        with open(path_to_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return DatasetData(
            tickers=data["tickers"],
            start_date=date.fromisoformat(data["start_date"]),
            end_date=date.fromisoformat(data["end_date"]),
        )

    @staticmethod
    def decode_experiment(path_to_json_file: Path) -> ExperimentConfig:
        """Decode the experiment config .json file to an experiment config model.

        Args:
            path_to_json_file (Path): Path to config file.

        Returns:
            ExperimentConfig: Filter choice, batching numbers and label options.
        """
        logger.debug("Decoding experiment .json config file.")
        with open(path_to_json_file, "r", encoding="utf-8") as f:
            experiment = json.load(f)["experiment"]

        batching = experiment["batching"]
        evaluation = experiment["evaluation"]
        return ExperimentConfig(
            filter_name=experiment["filter"],
            batch_size=int(batching["batch_size"]),
            top_k=int(batching["top_k"]),
            seed=int(batching["random_seed"]),
            scalability_sizes=[int(size) for size in evaluation["scalability_sizes"]],
            subtract_market=bool(evaluation.get("subtract_market", False)),
            standardize=bool(evaluation.get("standardize", False)),
        )
