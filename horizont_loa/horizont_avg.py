from collections import defaultdict

by_mode = defaultdict(list)

with open("horizont.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line or "->" not in line:
            continue

        _, mode, value = line.split("->")

        mode = mode.strip().lower()
        value = float(value.strip())

        by_mode[mode].append(value)


for mode in ["1_day", "7_days"]:
    if mode not in by_mode:
        continue

    values = by_mode[mode]
    average = sum(values) / len(values)

    print(f"{mode} -> {average}")