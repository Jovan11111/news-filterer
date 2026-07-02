import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.comparison_visualizer import ComparisonVisualizer  # noqa: E402
from src.dataset.news_sampler import NewsSampler  # noqa: E402
from src.evaluation.filter_evaluator import FilterEvaluator  # noqa: E402
from src.filtering.random_filter import RandomFilter  # noqa: E402
from src.models.evaluation_result import EvaluationResult  # noqa: E402
from src.utils.enums.filter_mode import FilterMode  # noqa: E402
from src.utils.enums.price_horizon import PriceHorizon  # noqa: E402
from src.utils.json_decoder import JsonDecoder  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_CONFIG_PATH = ROOT.joinpath("config", "experiment_config.json")
DATA_DIR = ROOT.joinpath("data")
FIGURES_DIR = ROOT.joinpath("results", "figures")
TABLES_DIR = ROOT.joinpath("results", "tables")

CSV_HEADER = (
    "dataset", "filter", "batch_size", "top_k", "mode", "horizon",
    "filter_value", "ideal_value", "average_value", "worst_value",
    "gap_to_ideal", "lift_over_average", "normalized_score",
)


def _format_cell(value: float | None) -> str:
    """Render an optional float for the console table."""
    return "n/a" if value is None else f"{value:+.3f}"


def _print_result(dataset_name: str, result: EvaluationResult):
    """Print a per-mode/horizon comparison table for one dataset run."""
    print(f"\n=== {result.filter_name} on {dataset_name} (batch {result.batch_size}, top {result.top_k}) ===")
    print(f"{'mode':>7} {'horizon':>8} {'filter':>9} {'ideal':>9} {'average':>9} {'worst':>9} {'gap->ideal':>11} {'lift>avg':>9} {'norm':>7}")
    for mode in FilterMode:
        for horizon in PriceHorizon:
            score = result.score(mode, horizon)
            if score is None:
                continue
            print(f"{mode.value:>7} {horizon.value:>8} {_format_cell(score.filter_value):>9} {_format_cell(score.ideal_value):>9} {_format_cell(score.average_value):>9} {_format_cell(score.worst_value):>9} {_format_cell(score.gap_to_ideal):>11} {_format_cell(score.lift_over_average):>9} {_format_cell(score.normalized_score):>7}")


def _write_rows(writer, dataset_name: str, result: EvaluationResult):
    """Append every score of one run to the open CSV writer."""
    for mode in FilterMode:
        for horizon in PriceHorizon:
            score = result.score(mode, horizon)
            if score is None:
                continue
            writer.writerow([
                dataset_name, result.filter_name, result.batch_size, result.top_k,
                mode.value, horizon.value,
                score.filter_value, score.ideal_value, score.average_value, score.worst_value,
                score.gap_to_ideal, score.lift_over_average, score.normalized_score,
            ])


def main():
    """Run the configured filter over every dataset and write scores and figures."""
    config = JsonDecoder.decode_experiment(EXPERIMENT_CONFIG_PATH)
    batch_size = config.batch_size
    top_k = config.top_k
    seed = config.seed
    subtract_market = config.subtract_market

    datasets = sorted(DATA_DIR.glob("*.db"))
    if not datasets:
        raise SystemExit(f"No dataset databases found in {DATA_DIR}.")

    evaluator = FilterEvaluator(top_k=top_k, subtract_market=subtract_market)
    visualizer = ComparisonVisualizer(output_dir=FIGURES_DIR)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR.joinpath("filter_news.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)

        for dataset in datasets:
            logger.info(f"Running filter on dataset {dataset.name}.")
            sampler = NewsSampler(database=dataset, batch_size=batch_size, seed=seed)
            batch = sampler.sample()
            if not batch:
                logger.warning(f"Dataset {dataset.name} yielded no sampled news; skipping.")
                continue

            news_filter = RandomFilter(top_k=top_k, seed=seed)
            result = evaluator.evaluate(news_filter, batch)

            _print_result(dataset.stem, result)
            _write_rows(writer, dataset.stem, result)
            visualizer.render(result, tag=dataset.stem)

    print(f"\nWrote scores to: {csv_path}")
    print(f"Wrote figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
