from pathlib import Path
import json
import statistics

ROOT = Path("./results/reports/vader")

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue

    summary_path = folder / "summary.json"

    if not summary_path.exists():
        print(f"{folder.name} -> summary.json nije pronađen")
        continue

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        per_cell = summary.get("per_cell", {})

        std_values = []

        for cell in per_cell.values():
            normalized_score = cell.get("normalized_score", {})
            std = normalized_score.get("std")

            if std is not None:
                std_values.append(std)

        if not std_values:
            print(f"{folder.name} -> nema std vrednosti")
            continue

        average_std = statistics.fmean(std_values)

        print(f"{folder.name} -> {average_std}")

    except (json.JSONDecodeError, OSError) as e:
        print(f"{folder.name} -> greška: {e}")