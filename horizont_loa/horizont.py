import json
from pathlib import Path
from statistics import fmean

ROOT_FOLDER = Path("./results/reports/")

for filter_folder in sorted(ROOT_FOLDER.iterdir()):
    if not filter_folder.is_dir():
        continue
    print(filter_folder)
    for folder in sorted(filter_folder.iterdir()):
        if not folder.is_dir():
            continue

        summary_file = folder / "summary.json"

        if not summary_file.exists():
            continue

        try:
            with summary_file.open("r", encoding="utf-8") as f:
                summary = json.load(f)

            per_cell = summary["per_cell"]

            for mode in ["1_day", "7_days"]:
                lift_values = []

                for horizon in ["buy", "sell", "affect"]:
                    key = f"{horizon}|{mode}"

                    value = per_cell[key]["lift_over_average"]["mean"]

                    if value is not None:
                        lift_values.append(value)

                if lift_values:
                    mean_lift = fmean(lift_values)
                    print(f"{folder.name} -> {mode} -> {mean_lift}")

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"{folder.name} -> ERROR: {e}")