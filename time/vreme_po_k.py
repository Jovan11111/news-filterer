import matplotlib.pyplot as plt

top_k = [5, 10, 20, 50, 100]

data = {
    "FinBERT": [
234.01904285714286,
234.3046142857143,
233.67997142857143,
271.2624666666667,
321.40816
    ],
    "Mini": [
138.39293125,
157.20124375,
215.37484999999998,
311.5357125,
220.20436
    ],
    "Premium AI": [
183.27375,
184.7998625,
310.1861375,
469.9454625,
426.98722
    ],
    "Random": [
0.0,
0.00005714285714285715,
0.0001,
0.0001,
0.00012000000000000002
    ],
    "Rule-based": [
0.11032857142857141,
0.10962857142857144,
0.1099,
0.12901666666666667,
0.14868
    ],
    "Trained model": [
0.5591571428571428,
0.5665428571428571,
0.5880285714285713,
0.68645,
0.78212
    ],
    "VADER": [
1.2249714285714286,
1.2145285714285714,
1.2251285714285713,
1.4273500000000001,
1.66542
    ]
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

plt.xlabel("Veličina izlaznog skupa (top_k)")
plt.ylabel("Vreme izvršavanja (sekundame)")
plt.title("Vreme izvršavanja po veličini izlaznog skupa")

plt.xticks(x, top_k)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("vreme_po_k.png", dpi=300)
plt.show()