import matplotlib.pyplot as plt

top_k = [50, 100, 200, 500, 1000, 2000, 5000]

data = {
    "FinBERT": [
9.340366666666666,
19.219175,
37.24692,
95.25175999999999,
188.72874,
370.75372000000004,
916.7674999999999
    ],
    "FinBERT cached": [
3.2621,
6.742125,
12.61484,
31.9474,
62.738519999999994,
123.37116,
306.24502
    ],
}

plt.figure(figsize=(10, 6))

x = range(len(top_k))

for filter_name, values in data.items():
    plt.plot(
        x,
        values,
        marker="o",
        label=filter_name
    )

plt.xlabel("Veličina ulaznog skupa (batch_size)")
plt.ylabel("Vreme izvršavanja (sekundame)")
plt.title("Vreme izvršavanja po veličini ulaznog skupa")

plt.xticks(x, top_k)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("vreme_po_bs_cached.png", dpi=300)
plt.show()