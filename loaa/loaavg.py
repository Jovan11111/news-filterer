from collections import defaultdict

by_batch_size = defaultdict(list)
by_top_k = defaultdict(list)
with open("loa.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line or "->" not in line:
            continue

        name, value = line.split("->")
        name = name.strip()
        value = float(value.strip())

        # Format: top_k_batch_size_tournament
        parts = name.split("_")

        top_k = int(parts[0])
        batch_size = int(parts[1])

        by_batch_size[batch_size].append(value)
        by_top_k[top_k].append(value)
print("topk")
for top_k in sorted(by_top_k):
    values = by_top_k[top_k]
    average = sum(values) / len(values)

    print(f"{top_k} -> {average}")


print("batch size")
for batch_size in sorted(by_batch_size):
    values = by_batch_size[batch_size]
    average = sum(values) / len(values)

    print(f"{batch_size} -> {average}")