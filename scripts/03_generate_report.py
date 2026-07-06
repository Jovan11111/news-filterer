import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.report_generator import ReportGenerator
from src.evaluation.filter_runner import FilterRunner
from src.filtering.filter import Filter
from src.filtering.finbert_filter import FinbertFilter
from src.filtering.llm_filter import LlmFilter
from src.filtering.random_filter import RandomFilter
from src.filtering.rule_based_filter import RuleBasedFilter
from src.filtering.trained_model_filter import TrainedModelFilter
from src.filtering.vader_sentiment_filter import VaderSentimentFilter
from src.utils.json_decoder import JsonDecoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Maps a config filter name to a factory that builds a fresh filter for each batch-size run.
FILTERS: dict[str, Callable[[int, int], Filter]] = {
    "random": lambda top_k, seed: RandomFilter(top_k=top_k, seed=seed),
    "rule_based": lambda top_k, seed: RuleBasedFilter(top_k=top_k),
    "vader": lambda top_k, seed: VaderSentimentFilter(top_k=top_k),
    "finbert": lambda top_k, seed: FinbertFilter(top_k=top_k),
    "finbert_cached": lambda top_k, seed: FinbertFilter(top_k=top_k, is_cached=True),
    "trained_model": lambda top_k, seed: TrainedModelFilter(top_k=top_k),
    "llm": lambda top_k, seed: LlmFilter(top_k=top_k, seed=seed),
    "llm_tournament": lambda top_k, seed: LlmFilter(top_k=top_k, seed=seed, is_tournament=True),
    "llm_premium": lambda top_k, seed: LlmFilter(top_k=top_k, seed=seed, is_premium=True),
    "llm_premium_tournament": lambda top_k, seed: LlmFilter(top_k=top_k, seed=seed, is_premium=True, is_tournament=True),
}

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_CONFIG_PATH = ROOT.joinpath("config", "experiment_config.json")
DATA_DIR = ROOT.joinpath("data")
GRAPHS_DIR = ROOT.joinpath("graphs")
REPORTS_DIR = ROOT.joinpath("results", "reports")


def _row_count(dataset: Path) -> int:
    """Return the number of news rows in a dataset database."""
    connection = sqlite3.connect(str(dataset))
    try:
        return connection.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    finally:
        connection.close()


def _pick_dataset() -> Path:
    """Pick the dataset with the most rows so large batch sizes are reachable."""
    datasets = sorted(DATA_DIR.glob("*.db"))
    if not datasets:
        raise SystemExit(f"No dataset databases found in {DATA_DIR}.")
    return max(datasets, key=_row_count)


def main():
    """Run the configured filter across batch sizes and write the graphs and PDF report."""
    config = JsonDecoder.decode_experiment(EXPERIMENT_CONFIG_PATH)
    if config.filter_name not in FILTERS:
        raise SystemExit(f"Unknown filter '{config.filter_name}'. Choose one of: {', '.join(sorted(FILTERS))}.")
    if config.top_k <= 0:
        raise SystemExit(f"top_k must be positive, got {config.top_k}.")

    top_k = config.top_k
    seed = config.seed
    batch_sizes = config.scalability_sizes
    subtract_market = config.subtract_market
    standardize = config.standardize

    dataset = _pick_dataset()
    available = _row_count(dataset)
    logger.info(f"Using dataset {dataset.name} with {available} rows for batch sizes {batch_sizes}.")

    factory = FILTERS[config.filter_name]
    label = "market-adjusted" if subtract_market else "raw"
    if standardize:
        label += ", z-scored"
    logger.info(f"Running filter '{config.filter_name}' with top_k={top_k} on {label} returns.")

    runner = FilterRunner(dataset=dataset, top_k=top_k, seed=seed, subtract_market=subtract_market, standardize=standardize)
    runs = runner.run(lambda: factory(top_k, seed), batch_sizes)

    generator = ReportGenerator(graphs_dir=GRAPHS_DIR, reports_dir=REPORTS_DIR)
    pdf_path = generator.generate(dataset_name=dataset.stem, top_k=top_k, seed=seed, runs=runs, subtract_market=subtract_market, standardize=standardize)

    print(f"\nWrote graphs to: {GRAPHS_DIR}")
    print(f"Wrote PDF report to: {pdf_path}")


if __name__ == "__main__":
    main()
