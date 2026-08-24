from collections import defaultdict

by_mode = defaultdict(list)

with open("rezim.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line or "->" not in line:
            continue

        _, mode, value = line.split("->")

        mode = mode.strip().lower()
        value = float(value.strip())

        by_mode[mode].append(value)


for mode in ["buy", "sell", "affect"]:
    if mode not in by_mode:
        continue

    values = by_mode[mode]
    average = sum(values) / len(values)

    print(f"{mode} -> {average*100}")