import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import tickers as ticker_lists  # noqa: E402
from src.dataset.dataset_builder import DatasetBuilder  # noqa: E402
from src.utils.json_decoder import JsonDecoder  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent.parent.joinpath("config", "dataset_config.json")


def main():
    """Build the dataset described by config/dataset_config.json and print its path."""
    dataset_config = JsonDecoder.decode(CONFIG_PATH)

    try:
        tickers = getattr(ticker_lists, dataset_config.tickers)
    except AttributeError:
        raise SystemExit(f"Unknown ticker list '{dataset_config.tickers}'. Try SP100, SP50, SP20, SP10.")

    builder = DatasetBuilder(
        tickers=tickers,
        start_date=dataset_config.start_date,
        end_date=dataset_config.end_date,
    )
    db_path = builder.build()
    print(f"Dataset written to: {db_path}")


if __name__ == "__main__":
    main()
