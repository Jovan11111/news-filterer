"""Non-LLM filter experiment runner (mirrors the premium/mini LLM sweeps).

Runs every non-LLM filter over the SAME (top_k x batch_size) matrix and the SAME
repeated-run protocol used by scripts/run_premium_experiments.py, so the results
drop straight into the same comparison. There is no tournament axis here (only
the LLM filter has one) and no token/cost accounting (these filters make no API
calls), but the per-run records, per-variant summaries and CSVs use the exact
same schema as the LLM sweeps -- the token/cost fields are simply zero -- so the
folders are directly comparable.

Timing IS measured for every filter: `filter_elapsed_seconds` is the pure
selection time (the time inside Filter.select summed over the three modes, as
reported by the evaluator) and `wall_seconds` is the whole run including
sampling. Model loading for FinBERT / the trained model happens in the filter's
constructor, OUTSIDE the timed region, so it never inflates the filter time.

Seeds match the LLM sweeps exactly (seed = BASE_SEED + run_index), so for a given
(top_k, batch_size, run_index) every filter -- including the LLM ones -- is
evaluated on the IDENTICAL sampled batch. Comparisons are therefore paired.

Filters covered (each written to its own folder under results/reports/):

    random          seeded random floor (stochastic; seed varied per run)
    rule_based      bullish-minus-bearish keyword count (deterministic)
    vader           VADER lexicon compound sentiment (deterministic)
    finbert         FinBERT sentiment, uncached (deterministic)
    finbert_cached  FinBERT sentiment, per-article cache reused across modes
    trained_model   project-trained TF-IDF -> SVD -> MLP regressor (deterministic)

The only per-filter "condition" besides top_k (the analog of the LLM's
tournament flag) is FinBERT's is_cached flag; it does not change WHICH articles
are picked, only how fast they are picked, so both variants are benchmarked as
two separate filters to expose that operational difference.

Note: deterministic filters still show run-to-run variance because each run
samples a different batch (the seed changes per run), so 10 runs give a genuine
distribution to average, exactly like the LLM sweeps.

Output layout (under results/reports/<filter>/):

    <filter>/
        run.log                     human-readable progress log
        runs.csv                    one row per run x mode x horizon (flat)
        manifest.csv                one row per variant (aggregates)
        <top_k>_<batch>_false/      one folder per variant
            run_01.json .. run_10.json
            summary.json            per-variant aggregates (mean/std over runs)

The script is resumable: a run whose run_XX.json already exists (status "ok") is
skipped, so it can be stopped and relaunched without losing work.

Run with the project venv (all filters, cheapest first):
    .\\venv\\Scripts\\python.exe -u scripts\\run_baseline_experiments.py

Or restrict to specific filters:
    .\\venv\\Scripts\\python.exe -u scripts\\run_baseline_experiments.py vader rule_based

Set BASELINE_EXP_SMOKE=1 to run a tiny validation matrix instead of the full one.
"""

import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.news_sampler import NewsSampler
from src.evaluation.filter_evaluator import FilterEvaluator
from src.filtering.finbert_filter import FinbertFilter
from src.filtering.random_filter import RandomFilter
from src.filtering.rule_based_filter import RuleBasedFilter
from src.filtering.trained_model_filter import TrainedModelFilter
from src.filtering.vader_sentiment_filter import VaderSentimentFilter
from src.utils.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.joinpath("data")
REPORTS_DIR = ROOT.joinpath("results", "reports")

# How many times each variant is run so its scores can be averaged. Each run
# uses a distinct seed, so the sampled batch differs, giving genuine run-to-run
# variance to average over -- even for deterministic filters.
RUNS_PER_VARIANT = 10
BASE_SEED = 1000

# Evaluation label settings: raw (unadjusted) returns, matching the LLM sweeps.
SUBTRACT_MARKET = False
STANDARDIZE = False

# --- Variant matrix -----------------------------------------------------------
# Every (top_k, batch_size) combination where the selection is non-degenerate
# (batch_size > top_k). This is a superset of the LLM sweeps' (top_k, batch_size)
# coverage, so every LLM variant has a matching baseline variant to compare with.
# There is no tournament axis: these filters rank the whole batch directly.

TOP_KS = [5, 10, 20, 50, 100]
BATCH_SIZES = [50, 100, 200, 500, 1000, 2000, 5000]

SMOKE_TOP_KS = [10]
SMOKE_BATCH_SIZES = [30, 60]
SMOKE_RUNS_PER_VARIANT = 2


# Filter registry: name -> factory(top_k, seed) -> Filter. Ordered cheapest-first
# so fast filters finish (and produce comparable results) before FinBERT's slow
# uncached CPU inference. is_cached is FinBERT's only extra condition (the analog
# of the LLM tournament flag); the two variants are benchmarked separately.
FILTERS: dict = {
    "random": lambda top_k, seed: RandomFilter(top_k=top_k, seed=seed),
    "rule_based": lambda top_k, seed: RuleBasedFilter(top_k=top_k),
    "vader": lambda top_k, seed: VaderSentimentFilter(top_k=top_k),
    "trained_model": lambda top_k, seed: TrainedModelFilter(top_k=top_k),
    "finbert_cached": lambda top_k, seed: FinbertFilter(top_k=top_k, is_cached=True),
    "finbert": lambda top_k, seed: FinbertFilter(top_k=top_k, is_cached=False),
}


def _row_count(dataset: Path) -> int:
    """Return the number of news rows in a dataset database."""
    import sqlite3

    connection = sqlite3.connect(str(dataset))
    try:
        return connection.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    finally:
        connection.close()


def _pick_dataset() -> Path:
    """Pick the dataset with the most rows (matches the LLM sweeps' choice)."""
    datasets = sorted(DATA_DIR.glob("*.db"))
    if not datasets:
        raise SystemExit(f"No dataset databases found in {DATA_DIR}.")
    return max(datasets, key=_row_count)


def _variant_name(top_k: int, batch_size: int) -> str:
    """Build the folder name for a variant: top_k_batch_false (no tournament)."""
    return f"{top_k}_{batch_size}_false"


def _build_variants(smoke: bool) -> list[dict]:
    """Assemble every non-degenerate (top_k x batch_size) variant, cheapest-first."""
    top_ks = SMOKE_TOP_KS if smoke else TOP_KS
    batch_sizes = SMOKE_BATCH_SIZES if smoke else BATCH_SIZES

    variants: list[dict] = []
    for top_k in top_ks:
        for batch_size in batch_sizes:
            if batch_size <= top_k:
                continue  # degenerate: cannot select top_k from a smaller batch
            variants.append(
                {
                    "top_k": top_k,
                    "batch_size": batch_size,
                    "tournament": False,
                    "group_size": None,
                    "name": _variant_name(top_k, batch_size),
                }
            )

    # Cheapest-first: small batches (and small top_k) run before the heavy ones.
    variants.sort(key=lambda v: (v["batch_size"], v["top_k"]))
    return variants


def _mean_std(values: list) -> dict:
    """Return the mean, sample stdev and count of a list of numbers."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(clean)}


def _run_once(filter_name: str, variant: dict, seed: int, run_index: int, dataset: Path) -> dict:
    """Sample a batch, evaluate one filter once, and return the full run record."""
    wall_start = time.perf_counter()
    sampler = NewsSampler(database=dataset, batch_size=variant["batch_size"], min_gap_days=0, seed=seed)
    batch = sampler.sample()
    if not batch:
        raise RuntimeError(f"Empty sample for batch_size {variant['batch_size']}.")

    news_filter = FILTERS[filter_name](variant["top_k"], seed)
    evaluator = FilterEvaluator(variant["top_k"], SUBTRACT_MARKET, STANDARDIZE)
    evaluation = evaluator.evaluate(news_filter, batch)
    wall_seconds = time.perf_counter() - wall_start

    run_meta = {
        "status": "ok",
        "actual_batch_size": evaluation.batch_size,
        # No API calls / tokens / cost for local filters; kept for schema parity.
        "api_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "filter_elapsed_seconds": evaluation.elapsed_seconds,
        "wall_seconds": wall_seconds,
    }

    record = {
        "variant": variant["name"],
        "top_k": variant["top_k"],
        "batch_size": variant["batch_size"],
        "tournament": variant["tournament"],
        "group_size": variant["group_size"],
        "model": news_filter.name,
        "subtract_market": SUBTRACT_MARKET,
        "standardize": STANDARDIZE,
        "seed": seed,
        "run_index": run_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pricing": {"input_per_1m": 0.0, "output_per_1m": 0.0},
        **run_meta,
        "scores": evaluation.to_dict()["scores"],
    }
    return record


def _error_record(filter_name: str, variant: dict, seed: int, run_index: int, error: Exception) -> dict:
    """Build the record stored when a run fails after all retries."""
    return {
        "variant": variant["name"],
        "top_k": variant["top_k"],
        "batch_size": variant["batch_size"],
        "tournament": variant["tournament"],
        "group_size": variant["group_size"],
        "model": filter_name,
        "subtract_market": SUBTRACT_MARKET,
        "standardize": STANDARDIZE,
        "seed": seed,
        "run_index": run_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pricing": {"input_per_1m": 0.0, "output_per_1m": 0.0},
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "actual_batch_size": 0,
        "api_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "filter_elapsed_seconds": 0.0,
        "wall_seconds": 0.0,
        "scores": [],
    }


def _run_with_retry(filter_name: str, variant: dict, seed: int, run_index: int, dataset: Path, attempts: int = 3) -> dict:
    """Run once, retrying transient failures a few times before recording an error."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return _run_once(filter_name, variant, seed, run_index, dataset)
        except Exception as exc:  # noqa: BLE001 - a single failure must not kill the sweep
            last_error = exc
            logger.warning(
                f"{filter_name}/{variant['name']} run {run_index} attempt {attempt}/{attempts} "
                f"failed: {type(exc).__name__}: {exc}"
            )
            time.sleep(min(30, 5 * attempt))
    return _error_record(filter_name, variant, seed, run_index, last_error)


def _aggregate_variant(variant: dict, records: list[dict]) -> dict:
    """Aggregate a variant's runs into mean/std per mode-horizon plus timing totals."""
    ok_records = [r for r in records if r.get("status") == "ok"]
    metrics = ("normalized_score", "lift_over_average", "filter_value", "gap_to_ideal")
    per_cell: dict = {}
    all_normalized: list = []
    for record in ok_records:
        for score in record["scores"]:
            key = f"{score['mode']}|{score['horizon']}"
            per_cell.setdefault(key, {metric: [] for metric in metrics})
            for metric in metrics:
                per_cell[key][metric].append(score.get(metric))
            if score.get("normalized_score") is not None:
                all_normalized.append(score["normalized_score"])

    per_cell_stats = {key: {metric: _mean_std(vals) for metric, vals in cell.items()} for key, cell in per_cell.items()}

    totals = {
        "runs_completed": len(ok_records),
        "runs_failed": len(records) - len(ok_records),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
        "estimated_cost_usd": 0.0,
        "filter_seconds": round(sum(r["filter_elapsed_seconds"] for r in ok_records), 3),
        "wall_seconds": round(sum(r["wall_seconds"] for r in ok_records), 3),
        "mean_wall_seconds": round(statistics.fmean([r["wall_seconds"] for r in ok_records]), 3) if ok_records else None,
        "mean_filter_seconds": round(statistics.fmean([r["filter_elapsed_seconds"] for r in ok_records]), 3) if ok_records else None,
    }

    return {
        "variant": variant["name"],
        "top_k": variant["top_k"],
        "batch_size": variant["batch_size"],
        "tournament": variant["tournament"],
        "group_size": variant["group_size"],
        "model": next((r.get("model") for r in ok_records if r.get("model")), None),
        "subtract_market": SUBTRACT_MARKET,
        "standardize": STANDARDIZE,
        "pricing": {"input_per_1m": 0.0, "output_per_1m": 0.0},
        "totals": totals,
        "overall_mean_normalized_score": statistics.fmean(all_normalized) if all_normalized else None,
        "per_cell": per_cell_stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _flatten_record(variant: dict, record: dict) -> list[dict]:
    """Flatten a stored run record into per-mode/horizon CSV rows."""
    rows: list[dict] = []
    for score in record["scores"]:
        rows.append(
            {
                "variant": variant["name"],
                "top_k": variant["top_k"],
                "batch_size": variant["batch_size"],
                "tournament": variant["tournament"],
                "group_size": variant["group_size"],
                "seed": record["seed"],
                "run_index": record["run_index"],
                "status": record["status"],
                "actual_batch_size": record["actual_batch_size"],
                "mode": score["mode"],
                "horizon": score["horizon"],
                "filter_value": score["filter_value"],
                "ideal_value": score["ideal_value"],
                "average_value": score["average_value"],
                "worst_value": score["worst_value"],
                "gap_to_ideal": score["gap_to_ideal"],
                "lift_over_average": score["lift_over_average"],
                "normalized_score": score["normalized_score"],
                "api_calls": record["api_calls"],
                "prompt_tokens": record["prompt_tokens"],
                "completion_tokens": record["completion_tokens"],
                "total_tokens": record["total_tokens"],
                "estimated_cost_usd": record["estimated_cost_usd"],
                "filter_elapsed_seconds": record["filter_elapsed_seconds"],
                "wall_seconds": record["wall_seconds"],
            }
        )
    return rows


def _manifest_row(summary: dict) -> dict:
    """Build one manifest.csv row from a variant summary."""
    totals = summary["totals"]
    return {
        "variant": summary["variant"],
        "top_k": summary["top_k"],
        "batch_size": summary["batch_size"],
        "tournament": summary["tournament"],
        "group_size": summary["group_size"],
        "model": summary["model"],
        "runs_completed": totals["runs_completed"],
        "runs_failed": totals["runs_failed"],
        "api_calls": totals["api_calls"],
        "total_tokens": totals["total_tokens"],
        "prompt_tokens": totals["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "estimated_cost_usd": totals["estimated_cost_usd"],
        "filter_seconds": totals["filter_seconds"],
        "mean_filter_seconds": totals["mean_filter_seconds"],
        "mean_wall_seconds": totals["mean_wall_seconds"],
        "overall_mean_normalized_score": summary["overall_mean_normalized_score"],
    }


def _load_records(variant_dir: Path) -> list[dict]:
    """Load every run_*.json record from a variant directory in run order."""
    records: list[dict] = []
    for run_file in sorted(variant_dir.glob("run_*.json")):
        try:
            records.append(json.loads(run_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Could not read {run_file}: {exc}")
    return records


def _write_json(path: Path, payload: dict) -> None:
    """Write a dict to a JSON file (pretty, UTF-8), creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _variant_from_record(record: dict) -> dict:
    """Reconstruct the minimal variant dict a rebuild needs from a stored run."""
    return {
        "name": record["variant"],
        "top_k": record["top_k"],
        "batch_size": record["batch_size"],
        "tournament": record["tournament"],
        "group_size": record["group_size"],
    }


def _rebuild_csvs(output_dir: Path) -> None:
    """Rebuild runs.csv, manifest.csv and summaries from every variant dir on disk."""
    run_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for variant_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        records = _load_records(variant_dir)
        if not records:
            continue
        variant = _variant_from_record(records[0])
        for record in records:
            if record.get("status") == "ok":
                run_rows.extend(_flatten_record(variant, record))
        summary = _aggregate_variant(variant, records)
        _write_json(variant_dir.joinpath("summary.json"), summary)
        manifest_rows.append(_manifest_row(summary))

    if run_rows:
        with output_dir.joinpath("runs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_rows)
    if manifest_rows:
        with output_dir.joinpath("manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)


def _run_filter(filter_name: str, variants: list[dict], runs_per_variant: int, dataset: Path) -> None:
    """Run the full variant matrix for one filter into results/reports/<filter>/."""
    output_dir = REPORTS_DIR.joinpath(filter_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(variants) * runs_per_variant
    logger.info(f"=== Filter '{filter_name}': {len(variants)} variants x {runs_per_variant} runs = {total_runs} runs. Output {output_dir}.")

    started = time.perf_counter()
    completed = 0
    for variant_index, variant in enumerate(variants, start=1):
        variant_dir = output_dir.joinpath(variant["name"])
        variant_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{filter_name}] variant {variant_index}/{len(variants)} {variant['name']}.")

        for run_index in range(1, runs_per_variant + 1):
            run_path = variant_dir.joinpath(f"run_{run_index:02d}.json")
            if run_path.exists():
                try:
                    if json.loads(run_path.read_text(encoding="utf-8")).get("status") == "ok":
                        completed += 1
                        continue
                except (json.JSONDecodeError, OSError):
                    pass

            seed = BASE_SEED + run_index
            record = _run_with_retry(filter_name, variant, seed, run_index, dataset)
            _write_json(run_path, record)
            completed += 1

            elapsed = time.perf_counter() - started
            rate = elapsed / completed if completed else 0.0
            remaining = (total_runs - completed) * rate
            logger.info(
                f"  [{filter_name}] run {run_index}/{runs_per_variant} [{record['status']}] "
                f"filter={record['filter_elapsed_seconds']:.2f}s wall={record['wall_seconds']:.2f}s "
                f"| done {completed}/{total_runs} eta~{remaining / 3600:.2f}h"
            )
            _rebuild_csvs(output_dir)

    _rebuild_csvs(output_dir)
    total_elapsed = time.perf_counter() - started
    logger.info(f"=== Filter '{filter_name}' done: {completed}/{total_runs} runs in {total_elapsed / 3600:.2f}h. Output {output_dir}.")


def main() -> None:
    """Run every requested filter over the full variant matrix, cheapest filter first."""
    smoke = os.getenv("BASELINE_EXP_SMOKE") == "1"
    runs_per_variant = SMOKE_RUNS_PER_VARIANT if smoke else RUNS_PER_VARIANT

    requested = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    unknown = [name for name in requested if name not in FILTERS]
    if unknown:
        raise SystemExit(f"Unknown filter(s): {unknown}. Choose from {list(FILTERS)}.")
    filter_names = requested or list(FILTERS)

    dataset = _pick_dataset()
    available = _row_count(dataset)
    variants = _build_variants(smoke)

    logger.info(f"{'SMOKE ' if smoke else ''}Baseline experiments: filters={filter_names}.")
    logger.info(f"{len(variants)} variants x {runs_per_variant} runs each. Dataset {dataset.name} ({available} rows).")

    overall_start = time.perf_counter()
    for filter_name in filter_names:
        _run_filter(filter_name, variants, runs_per_variant, dataset)
    logger.info(f"All requested filters done in {(time.perf_counter() - overall_start) / 3600:.2f}h.")


if __name__ == "__main__":
    main()
