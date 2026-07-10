"""Overnight premium-LLM experiment runner.

Sweeps a large matrix of (top_k x batch_size x tournament x group_size) variants
for the premium LLM filter, running each variant many times so the results can
be averaged. For every run it records the full baseline-relative scores plus the
operational cost of the run: wall time, filter time, the number of model calls,
and the exact prompt/completion token counts (with an estimated dollar cost).

Output layout (under results/reports/premium AI/):

    premium AI/
        run.log                     human-readable progress log
        runs.csv                    one row per run x mode x horizon (flat)
        manifest.csv                one row per variant (aggregates)
        <top_k>_<batch>_<tf>[...]/  one folder per variant
            run_01.json .. run_NN.json
            summary.json            per-variant aggregates (mean/std over runs)

The script is resumable: a run whose run_XX.json already exists (status "ok") is
skipped, so it can be stopped and relaunched without losing work. Variants are
ordered cheapest-first so a broad spread of results accumulates early.

Pricing is an ASSUMPTION (real gpt-5.4 contract rates are unknown); the assumed
per-1M-token rates are stored in every summary so the dollar figures can be
recomputed from the recorded token counts at any time.

Run with the project venv:
    .\\venv\\Scripts\\python.exe scripts\\run_premium_experiments.py

Set PREMIUM_EXP_SMOKE=1 to run a tiny validation matrix instead of the full one.
"""

import json
import math
import os
import sqlite3
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.filter_evaluator import FilterEvaluator
from src.filtering.llm_filter import LlmFilter
from src.dataset.news_sampler import NewsSampler
from src.utils.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.joinpath("data")

# Output folder and model can be overridden via the environment so a second
# process (e.g. a cheaper model) can run in parallel into a separate folder
# without touching .env or disturbing another in-flight run. MODEL_OVERRIDE is
# passed explicitly to the filter (wins over PREMIUM_AI_MODEL); when unset the
# .env default is used.
OUTPUT_NAME = os.getenv("PREMIUM_EXP_OUTPUT", "premium AI")
OUTPUT_DIR = ROOT.joinpath("results", "reports", OUTPUT_NAME)
MODEL_OVERRIDE = os.getenv("PREMIUM_EXP_MODEL") or None

# How many times each variant is run so its scores can be averaged. Each run
# uses a distinct seed, so both the sampled batch and the model seed differ,
# giving genuine run-to-run variance to average over.
RUNS_PER_VARIANT = 10
BASE_SEED = 1000

# Evaluation label settings for this official set: raw (unadjusted) returns.
SUBTRACT_MARKET = False
STANDARDIZE = False

# ASSUMED gpt-5.4 pricing in USD per 1M tokens. Real contract rates are unknown;
# adjust these two numbers and the dollar costs recompute from the token counts.
PRICE_INPUT_PER_1M = 2.50
PRICE_OUTPUT_PER_1M = 10.00

# --- Variant matrix -----------------------------------------------------------

TOP_KS = [5, 10, 20]

# Batch sizes small enough to rank in a single prompt (no tournament needed).
NON_TOURNAMENT_BATCHES = [50, 100, 200, 500, 1000]

# (batch_size, [group_sizes]) for tournament variants. group_size must exceed
# top_k; every group size here is > the largest top_k so all top_k values apply.
TOURNAMENT_SPECS = [
    (100, [30, 50]),
    (200, [30, 50, 100]),
    (500, [50, 100]),
    (1000, [50, 100]),
    (2000, [50]),
    (5000, [50]),
]

SMOKE_TOP_KS = [10]
SMOKE_NON_TOURNAMENT_BATCHES = [30]
SMOKE_TOURNAMENT_SPECS = [(60, [30])]
SMOKE_RUNS_PER_VARIANT = 2


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


def _estimate_calls(batch_size: int, tournament: bool, group_size: int | None, top_k: int) -> int:
    """Estimate the model-call count for one selection, for cheapest-first ordering."""
    if batch_size <= top_k:
        return 0
    if not tournament:
        return 1
    calls = 0
    pool = batch_size
    while pool > group_size:
        groups = math.ceil(pool / group_size)
        calls += groups
        # Survivors mirror the real tournament: each group yields min(top_k, its
        # size), so a final partial group smaller than top_k passes through whole.
        survivors = 0
        remaining = pool
        for _ in range(groups):
            take = min(group_size, remaining)
            survivors += min(top_k, take)
            remaining -= take
        if survivors >= pool:
            break
        pool = survivors
    if pool > top_k:
        calls += 1
    return calls


def _variant_name(top_k: int, batch_size: int, tournament: bool, group_size: int | None) -> str:
    """Build the folder name for a variant: top_k_batch_size_tournament[_g<group>]."""
    name = f"{top_k}_{batch_size}_{str(tournament).lower()}"
    if tournament:
        name += f"_g{group_size}"
    return name


def _build_variants(smoke: bool) -> list[dict]:
    """Assemble every experiment variant, ordered cheapest-first (per run)."""
    top_ks = SMOKE_TOP_KS if smoke else TOP_KS
    non_tournament = SMOKE_NON_TOURNAMENT_BATCHES if smoke else NON_TOURNAMENT_BATCHES
    tournament_specs = SMOKE_TOURNAMENT_SPECS if smoke else TOURNAMENT_SPECS

    variants: list[dict] = []
    for top_k in top_ks:
        for batch_size in non_tournament:
            variants.append({"top_k": top_k, "batch_size": batch_size, "tournament": False, "group_size": None})
        for batch_size, group_sizes in tournament_specs:
            for group_size in group_sizes:
                if group_size <= top_k:
                    continue
                variants.append({"top_k": top_k, "batch_size": batch_size, "tournament": True, "group_size": group_size})

    for variant in variants:
        variant["name"] = _variant_name(variant["top_k"], variant["batch_size"], variant["tournament"], variant["group_size"])
        variant["est_calls_per_run"] = _estimate_calls(variant["batch_size"], variant["tournament"], variant["group_size"], variant["top_k"])

    variants.sort(key=lambda v: (v["est_calls_per_run"], v["batch_size"], v["top_k"]))
    return variants


def _estimated_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost of a set of tokens under the assumed rates."""
    return prompt_tokens / 1_000_000 * PRICE_INPUT_PER_1M + completion_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M


def _make_filter(variant: dict, seed: int) -> LlmFilter:
    """Build a fresh premium LlmFilter for one run from a variant spec."""
    return LlmFilter(
        top_k=variant["top_k"],
        is_premium=True,
        is_tournament=variant["tournament"],
        group_size=variant["group_size"] or 50,
        seed=seed,
        model=MODEL_OVERRIDE,
    )


def _run_once(variant: dict, seed: int, run_index: int, dataset: Path) -> dict:
    """Sample a batch, evaluate the filter once, and return the full run record."""
    wall_start = time.perf_counter()
    sampler = NewsSampler(database=dataset, batch_size=variant["batch_size"], min_gap_days=0, seed=seed)
    batch = sampler.sample()
    if not batch:
        raise RuntimeError(f"Empty sample for batch_size {variant['batch_size']}.")

    news_filter = _make_filter(variant, seed)
    evaluator = FilterEvaluator(variant["top_k"], SUBTRACT_MARKET, STANDARDIZE)
    evaluation = evaluator.evaluate(news_filter, batch)
    wall_seconds = time.perf_counter() - wall_start

    prompt_tokens = news_filter.prompt_tokens
    completion_tokens = news_filter.completion_tokens
    run_meta = {
        "status": "ok",
        "actual_batch_size": evaluation.batch_size,
        "api_calls": news_filter.api_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": news_filter.total_tokens,
        "estimated_cost_usd": round(_estimated_cost(prompt_tokens, completion_tokens), 6),
        "filter_elapsed_seconds": evaluation.elapsed_seconds,
        "wall_seconds": wall_seconds,
    }

    record = {
        "variant": variant["name"],
        "top_k": variant["top_k"],
        "batch_size": variant["batch_size"],
        "tournament": variant["tournament"],
        "group_size": variant["group_size"],
        "model": news_filter.model,
        "subtract_market": SUBTRACT_MARKET,
        "standardize": STANDARDIZE,
        "seed": seed,
        "run_index": run_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pricing": {"input_per_1m": PRICE_INPUT_PER_1M, "output_per_1m": PRICE_OUTPUT_PER_1M},
        **run_meta,
        "scores": evaluation.to_dict()["scores"],
    }
    return record


def _run_with_retry(variant: dict, seed: int, run_index: int, dataset: Path, attempts: int = 3) -> dict:
    """Run once, retrying transient failures a few times before recording an error."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return _run_once(variant, seed, run_index, dataset)
        except Exception as exc:  # noqa: BLE001 - overnight run must survive any single failure
            last_error = exc
            logger.warning(f"{variant['name']} run {run_index} attempt {attempt}/{attempts} failed: {type(exc).__name__}: {exc}")
            time.sleep(min(30, 5 * attempt))
    return {
        "variant": variant["name"],
        "top_k": variant["top_k"],
        "batch_size": variant["batch_size"],
        "tournament": variant["tournament"],
        "group_size": variant["group_size"],
        "model": None,
        "subtract_market": SUBTRACT_MARKET,
        "standardize": STANDARDIZE,
        "seed": seed,
        "run_index": run_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pricing": {"input_per_1m": PRICE_INPUT_PER_1M, "output_per_1m": PRICE_OUTPUT_PER_1M},
        "status": "error",
        "error": f"{type(last_error).__name__}: {last_error}",
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


def _mean_std(values: list[float]) -> dict:
    """Return the mean, sample stdev and count of a list of numbers."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(clean)}


def _aggregate_variant(variant: dict, records: list[dict]) -> dict:
    """Aggregate a variant's runs into mean/std per mode-horizon plus cost totals."""
    ok_records = [r for r in records if r.get("status") == "ok"]
    metrics = ("normalized_score", "lift_over_average", "filter_value", "gap_to_ideal")
    per_cell: dict[str, dict] = {}
    all_normalized: list[float] = []
    for record in ok_records:
        for score in record["scores"]:
            key = f"{score['mode']}|{score['horizon']}"
            per_cell.setdefault(key, {metric: [] for metric in metrics})
            for metric in metrics:
                per_cell[key][metric].append(score.get(metric))
            if score.get("normalized_score") is not None:
                all_normalized.append(score["normalized_score"])

    per_cell_stats = {key: {metric: _mean_std(vals) for metric, vals in cell.items()} for key, cell in per_cell.items()}

    total_prompt = sum(r["prompt_tokens"] for r in ok_records)
    total_completion = sum(r["completion_tokens"] for r in ok_records)
    totals = {
        "runs_completed": len(ok_records),
        "runs_failed": len(records) - len(ok_records),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "api_calls": sum(r["api_calls"] for r in ok_records),
        "estimated_cost_usd": round(_estimated_cost(total_prompt, total_completion), 6),
        "filter_seconds": round(sum(r["filter_elapsed_seconds"] for r in ok_records), 3),
        "wall_seconds": round(sum(r["wall_seconds"] for r in ok_records), 3),
        "mean_wall_seconds": round(statistics.fmean([r["wall_seconds"] for r in ok_records]), 3) if ok_records else None,
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
        "pricing": {"input_per_1m": PRICE_INPUT_PER_1M, "output_per_1m": PRICE_OUTPUT_PER_1M},
        "totals": totals,
        "overall_mean_normalized_score": statistics.fmean(all_normalized) if all_normalized else None,
        "per_cell": per_cell_stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
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


def _rebuild_csvs(all_variants: list[dict]) -> None:
    """Rebuild runs.csv and manifest.csv from every variant's per-run JSON files."""
    import csv

    run_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for variant in all_variants:
        variant_dir = OUTPUT_DIR.joinpath(variant["name"])
        if not variant_dir.exists():
            continue
        records = _load_records(variant_dir)
        if not records:
            continue
        for record in records:
            if record.get("status") == "ok":
                meta = {
                    "status": record["status"],
                    "actual_batch_size": record["actual_batch_size"],
                    "api_calls": record["api_calls"],
                    "prompt_tokens": record["prompt_tokens"],
                    "completion_tokens": record["completion_tokens"],
                    "total_tokens": record["total_tokens"],
                    "estimated_cost_usd": record["estimated_cost_usd"],
                    "filter_elapsed_seconds": record["filter_elapsed_seconds"],
                    "wall_seconds": record["wall_seconds"],
                }
                run_rows.extend(_flatten_record(variant, record, meta))
        summary = _aggregate_variant(variant, records)
        _write_json(variant_dir.joinpath("summary.json"), summary)
        manifest_rows.append(_manifest_row(summary))

    if run_rows:
        with OUTPUT_DIR.joinpath("runs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_rows)
    if manifest_rows:
        with OUTPUT_DIR.joinpath("manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)


def _flatten_record(variant: dict, record: dict, meta: dict) -> list[dict]:
    """Flatten a stored run record into per-mode/horizon CSV rows."""
    rows: list[dict] = []
    for score in record["scores"]:
        rows.append({
            "variant": variant["name"],
            "top_k": variant["top_k"],
            "batch_size": variant["batch_size"],
            "tournament": variant["tournament"],
            "group_size": variant["group_size"],
            "seed": record["seed"],
            "run_index": record["run_index"],
            "status": meta["status"],
            "actual_batch_size": meta["actual_batch_size"],
            "mode": score["mode"],
            "horizon": score["horizon"],
            "filter_value": score["filter_value"],
            "ideal_value": score["ideal_value"],
            "average_value": score["average_value"],
            "worst_value": score["worst_value"],
            "gap_to_ideal": score["gap_to_ideal"],
            "lift_over_average": score["lift_over_average"],
            "normalized_score": score["normalized_score"],
            "api_calls": meta["api_calls"],
            "prompt_tokens": meta["prompt_tokens"],
            "completion_tokens": meta["completion_tokens"],
            "total_tokens": meta["total_tokens"],
            "estimated_cost_usd": meta["estimated_cost_usd"],
            "filter_elapsed_seconds": meta["filter_elapsed_seconds"],
            "wall_seconds": meta["wall_seconds"],
        })
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
        "mean_wall_seconds": totals["mean_wall_seconds"],
        "overall_mean_normalized_score": summary["overall_mean_normalized_score"],
    }


def _variant_from_record(record: dict) -> dict:
    """Reconstruct the minimal variant dict a rebuild needs from a stored run."""
    return {
        "name": record["variant"],
        "top_k": record["top_k"],
        "batch_size": record["batch_size"],
        "tournament": record["tournament"],
        "group_size": record["group_size"],
    }


def _rebuild_csvs_from_disk() -> None:
    """Rebuild runs.csv, manifest.csv and summaries from EVERY variant dir on disk.

    Unlike _rebuild_csvs (scoped to one run's in-memory matrix), this scans the
    whole output folder, so results from several sweeps written into the same
    folder (e.g. a later large-top_k sweep) merge instead of overwriting.
    """
    import csv

    run_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for variant_dir in sorted(p for p in OUTPUT_DIR.iterdir() if p.is_dir()):
        records = _load_records(variant_dir)
        if not records:
            continue
        variant = _variant_from_record(records[0])
        for record in records:
            if record.get("status") == "ok":
                meta = {
                    "status": record["status"],
                    "actual_batch_size": record["actual_batch_size"],
                    "api_calls": record["api_calls"],
                    "prompt_tokens": record["prompt_tokens"],
                    "completion_tokens": record["completion_tokens"],
                    "total_tokens": record["total_tokens"],
                    "estimated_cost_usd": record["estimated_cost_usd"],
                    "filter_elapsed_seconds": record["filter_elapsed_seconds"],
                    "wall_seconds": record["wall_seconds"],
                }
                run_rows.extend(_flatten_record(variant, record, meta))
        summary = _aggregate_variant(variant, records)
        _write_json(variant_dir.joinpath("summary.json"), summary)
        manifest_rows.append(_manifest_row(summary))

    if run_rows:
        with OUTPUT_DIR.joinpath("runs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_rows)
    if manifest_rows:
        with OUTPUT_DIR.joinpath("manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)


def main():
    """Run the full premium experiment matrix, writing organized results as it goes."""
    smoke = os.getenv("PREMIUM_EXP_SMOKE") == "1"
    runs_per_variant = SMOKE_RUNS_PER_VARIANT if smoke else RUNS_PER_VARIANT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = _pick_dataset()
    available = _row_count(dataset)
    variants = _build_variants(smoke)

    total_runs = len(variants) * runs_per_variant
    logger.info(f"{'SMOKE ' if smoke else ''}Premium experiment: {len(variants)} variants x {runs_per_variant} runs = {total_runs} runs.")
    logger.info(f"Model override: {MODEL_OVERRIDE or '(env default)'}. Output under {OUTPUT_DIR}.")
    logger.info(f"Dataset {dataset.name} ({available} rows). Output under {OUTPUT_DIR}.")

    started = time.perf_counter()
    completed = 0
    for variant_index, variant in enumerate(variants, start=1):
        variant_dir = OUTPUT_DIR.joinpath(variant["name"])
        variant_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[variant {variant_index}/{len(variants)}] {variant['name']} (~{variant['est_calls_per_run']} calls/run).")

        for run_index in range(1, runs_per_variant + 1):
            run_path = variant_dir.joinpath(f"run_{run_index:02d}.json")
            if run_path.exists():
                try:
                    existing = json.loads(run_path.read_text(encoding="utf-8"))
                    if existing.get("status") == "ok":
                        completed += 1
                        continue
                except (json.JSONDecodeError, OSError):
                    pass

            seed = BASE_SEED + run_index
            record = _run_with_retry(variant, seed, run_index, dataset)
            _write_json(run_path, record)
            completed += 1

            elapsed = time.perf_counter() - started
            rate = elapsed / completed if completed else 0.0
            remaining = (total_runs - completed) * rate
            status = record["status"]
            tokens = record["total_tokens"]
            logger.info(
                f"  run {run_index}/{runs_per_variant} [{status}] "
                f"calls={record['api_calls']} tokens={tokens} "
                f"wall={record['wall_seconds']:.1f}s | done {completed}/{total_runs} "
                f"eta~{remaining / 3600:.1f}h"
            )
            _rebuild_csvs(variants)

    _rebuild_csvs(variants)
    total_elapsed = time.perf_counter() - started
    logger.info(f"Finished {completed}/{total_runs} runs in {total_elapsed / 3600:.2f}h. Results under {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
