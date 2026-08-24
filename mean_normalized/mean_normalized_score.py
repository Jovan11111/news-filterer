import json
from pathlib import Path

ROOT_FOLDER = Path("./results/reports/")
for filter in sorted(ROOT_FOLDER.iterdir()):
    if not filter.is_dir():
        continue
    print(filter)
    for folder in sorted(filter.iterdir()):
        if not folder.is_dir():
            continue

        summary_file = folder / "summary.json"

        if not summary_file.exists():
            continue

        try:
            with summary_file.open("r", encoding="utf-8") as f:
                summary = json.load(f)

            overall_mean_normalized_score = summary["overall_mean_normalized_score"]

            print(f"{folder.name} -> {overall_mean_normalized_score}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"{folder.name} -> ERROR: {e}")