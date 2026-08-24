from collections import defaultdict
values = []
with open("at.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line or "->" not in line:
            continue

        name, value = line.split("->")
        name = name.strip()
        value = float(value.strip())

        parts = name.split("_")

        values.append(value)

print(f"Overall average: {sum(values) / len(values) * 100}")
