import json
from pathlib import Path
from statistics import fmean

ROOT_FOLDER = Path("./results/reports/")

for filter_folder in sorted(ROOT_FOLDER.iterdir()):
    if not filter_folder.is_dir():
        continue

    print(filter_folder.name)

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

            lift_values = [
                cell["lift_over_average"]["mean"]
                for cell in per_cell.values()
                if cell["lift_over_average"]["mean"] is not None
            ]

            mean_lift_over_average = fmean(lift_values)

            print(f"{folder.name} -> {mean_lift_over_average}")

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"{folder.name} -> ERROR: {e}")