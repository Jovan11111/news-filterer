import argparse
import json
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.report_generator import ReportGenerator  # noqa: E402
from src.evaluation.filter_runner import FilterRunner  # noqa: E402
from src.filtering.filter import Filter  # noqa: E402
from src.filtering.finbert_filter import FinbertFilter  # noqa: E402
from src.filtering.llm_filter import LlmFilter  # noqa: E402
from src.filtering.random_filter import RandomFilter  # noqa: E402
from src.filtering.rule_based_filter import RuleBasedFilter  # noqa: E402
from src.filtering.vader_sentiment_filter import VaderSentimentFilter  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# Maps a CLI name to a factory that builds a fresh filter for each batch-size run.
FILTERS: dict[str, Callable[[int, int], Filter]] = {
    "random": lambda top_k, seed: RandomFilter(top_k=top_k, seed=seed),
    "rule_based": lambda top_k, seed: RuleBasedFilter(top_k=top_k),
    "vader": lambda top_k, seed: VaderSentimentFilter(top_k=top_k),
    "finbert": lambda top_k, seed: FinbertFilter(top_k=top_k),
    "finbert_cached": lambda top_k, seed: FinbertFilter(top_k=top_k, is_cached=True),
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


def load_config() -> dict:
    """Read the experiment configuration file."""
    with open(EXPERIMENT_CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)["experiment"]


def row_count(dataset: Path) -> int:
    """Return the number of news rows in a dataset database."""
    connection = sqlite3.connect(str(dataset))
    try:
        return connection.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    finally:
        connection.close()


def pick_dataset() -> Path:
    """Pick the dataset with the most rows so large batch sizes are reachable."""
    datasets = sorted(DATA_DIR.glob("*.db"))
    if not datasets:
        raise SystemExit(f"No dataset databases found in {DATA_DIR}.")
    return max(datasets, key=row_count)


def parse_args() -> argparse.Namespace:
    """Parse command-line options, chiefly which filter to evaluate."""
    parser = argparse.ArgumentParser(description="Generate a PDF report for one filter across batch sizes.")
    parser.add_argument("--filter", dest="filter_name", choices=sorted(FILTERS), default="random", help="Which filter to run (default: random).")
    parser.add_argument("--top-k", dest="top_k", type=int, default=None, help="How many articles each filter selects (default: config batching.top_k).")
    market = parser.add_mutually_exclusive_group()
    market.add_argument("--subtract-market", dest="subtract_market", action="store_true", default=None, help="Subtract the S&P 500 change (use market-adjusted/abnormal returns).")
    market.add_argument("--no-subtract-market", dest="subtract_market", action="store_false", help="Use raw price changes (overrides the config flag).")
    standardize = parser.add_mutually_exclusive_group()
    standardize.add_argument("--standardize", dest="standardize", action="store_true", default=None, help="Divide moves by 30-day volatility (use z-scored returns).")
    standardize.add_argument("--no-standardize", dest="standardize", action="store_false", help="Do not standardize (overrides the config flag).")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    top_k = args.top_k if args.top_k is not None else int(config["batching"]["top_k"])
    if top_k <= 0:
        raise SystemExit(f"top_k must be positive, got {top_k}.")
    seed = int(config["batching"]["random_seed"])
    batch_sizes = [int(size) for size in config["evaluation"]["scalability_sizes"]]
    config_subtract = bool(config["evaluation"].get("subtract_market", False))
    subtract_market = args.subtract_market if args.subtract_market is not None else config_subtract
    config_standardize = bool(config["evaluation"].get("standardize", False))
    standardize = args.standardize if args.standardize is not None else config_standardize

    dataset = pick_dataset()
    available = row_count(dataset)
    logger.info(f"Using dataset {dataset.name} with {available} rows for batch sizes {batch_sizes}.")

    factory = FILTERS[args.filter_name]
    label = "market-adjusted" if subtract_market else "raw"
    if standardize:
        label += ", z-scored"
    logger.info(f"Running filter '{args.filter_name}' with top_k={top_k} on {label} returns.")

    runner = FilterRunner(dataset=dataset, top_k=top_k, seed=seed, subtract_market=subtract_market, standardize=standardize)
    runs = runner.run(lambda: factory(top_k, seed), batch_sizes)

    generator = ReportGenerator(graphs_dir=GRAPHS_DIR, reports_dir=REPORTS_DIR)
    pdf_path = generator.generate(dataset_name=dataset.stem, top_k=top_k, seed=seed, runs=runs, subtract_market=subtract_market, standardize=standardize)

    print(f"\nWrote graphs to: {GRAPHS_DIR}")
    print(f"Wrote PDF report to: {pdf_path}")


if __name__ == "__main__":
    main()
