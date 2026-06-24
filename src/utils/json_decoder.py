import json
from datetime import date
from pathlib import Path

from src.models.dataset_data import DatasetData
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
