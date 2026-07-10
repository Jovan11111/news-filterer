"""Additional sweep: large selection sizes (top_k = 50 and 100).

Reuses the exact machinery of run_premium_experiments (same filter, evaluator,
sampler, per-run records, token/cost accounting) but for large top_k, and only
on batches big enough that selecting that many items is still discriminating
(picking the top 50-100 of thousands is a realistic "large selection" regime;
picking 100 of 100 is degenerate, so small batches are excluded).

Results are written into the SAME output folder as the base sweep (chosen via
PREMIUM_EXP_OUTPUT) and the CSVs are rebuilt by scanning the whole folder, so
these variants MERGE with the existing top_k in {5,10,20} results rather than
overwriting them. New per-variant folders follow the same naming convention.

Run this AFTER the base sweeps finish, once per model:

    # premium (gpt-5.4, model from .env)
    $env:PREMIUM_EXP_OUTPUT="premium AI"
    ./venv/Scripts/python.exe -u scripts/run_topk_large_experiments.py *>> "results/reports/premium AI/run.log"

    # mini (gpt-5.4-mini)
    $env:PREMIUM_EXP_MODEL="gpt-5.4-mini"; $env:PREMIUM_EXP_OUTPUT="mini"
    ./venv/Scripts/python.exe -u scripts/run_topk_large_experiments.py *>> "results/reports/mini/run.log"
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_premium_experiments as base
from run_premium_experiments import (
    BASE_SEED,
    MODEL_OVERRIDE,
    OUTPUT_DIR,
    RUNS_PER_VARIANT,
    _estimate_calls,
    _pick_dataset,
    _rebuild_csvs_from_disk,
    _row_count,
    _run_with_retry,
    _variant_name,
    _write_json,
)

# Large selection sizes to add.
TOP_KS = [50, 100]

# Non-tournament (whole batch in one prompt): only where batch is well above
# top_k so the selection is still discriminating, and small enough to fit one
# prompt. Keyed by top_k.
NON_TOURNAMENT_BATCHES = {
    50: [500, 1000],
    100: [500, 1000],
}

# Tournament specs (group_size MUST exceed top_k). Each entry: (top_k, [group
# sizes], [batch sizes]).
TOURNAMENT_SPECS = [
    (50, [100, 200], [1000, 2000, 5000]),
    (100, [200], [1000, 2000, 5000]),
]


def build_variants() -> list[dict]:
    """Assemble the large-top_k variants, ordered cheapest-first (per run)."""
    variants: list[dict] = []
    for top_k in TOP_KS:
        for batch_size in NON_TOURNAMENT_BATCHES[top_k]:
            variants.append({"top_k": top_k, "batch_size": batch_size, "tournament": False, "group_size": None})
    for top_k, group_sizes, batch_sizes in TOURNAMENT_SPECS:
        for group_size in group_sizes:
            for batch_size in batch_sizes:
                if group_size <= top_k:
                    continue
                variants.append({"top_k": top_k, "batch_size": batch_size, "tournament": True, "group_size": group_size})

    for variant in variants:
        variant["name"] = _variant_name(variant["top_k"], variant["batch_size"], variant["tournament"], variant["group_size"])
        variant["est_calls_per_run"] = _estimate_calls(variant["batch_size"], variant["tournament"], variant["group_size"], variant["top_k"])

    variants.sort(key=lambda v: (v["est_calls_per_run"], v["batch_size"], v["top_k"]))
    return variants


def main():
    """Run the large-top_k sweep, merging results into the shared output folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = _pick_dataset()
    available = _row_count(dataset)
    variants = build_variants()
    total_runs = len(variants) * RUNS_PER_VARIANT

    base.logger.info(f"Top-K LARGE sweep: {len(variants)} variants x {RUNS_PER_VARIANT} runs = {total_runs} runs.")
    base.logger.info(f"Model override: {MODEL_OVERRIDE or '(env default)'}. Output under {OUTPUT_DIR}.")
    base.logger.info(f"Dataset {dataset.name} ({available} rows).")

    started = time.perf_counter()
    completed = 0
    for variant_index, variant in enumerate(variants, start=1):
        variant_dir = OUTPUT_DIR.joinpath(variant["name"])
        variant_dir.mkdir(parents=True, exist_ok=True)
        base.logger.info(f"[variant {variant_index}/{len(variants)}] {variant['name']} (~{variant['est_calls_per_run']} calls/run).")

        for run_index in range(1, RUNS_PER_VARIANT + 1):
            run_path = variant_dir.joinpath(f"run_{run_index:02d}.json")
            if run_path.exists():
                try:
                    if json.loads(run_path.read_text(encoding="utf-8")).get("status") == "ok":
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
            base.logger.info(
                f"  run {run_index}/{RUNS_PER_VARIANT} [{record['status']}] "
                f"calls={record['api_calls']} tokens={record['total_tokens']} "
                f"wall={record['wall_seconds']:.1f}s | done {completed}/{total_runs} "
                f"eta~{remaining / 3600:.1f}h"
            )
            _rebuild_csvs_from_disk()

    _rebuild_csvs_from_disk()
    total_elapsed = time.perf_counter() - started
    base.logger.info(f"Top-K LARGE sweep done: {completed}/{total_runs} runs in {total_elapsed / 3600:.2f}h. Output {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
