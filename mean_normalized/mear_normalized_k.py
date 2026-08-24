import matplotlib.pyplot as plt

top_k = [5, 10, 20, 50, 100]

data = {
    "FinBERT": [
1.7755495183017949,
1.502261008911445,
1.283569722911188,
1.5555399522786333,
0.4739704135818803
    ],
    "Mini": [
1.7636135741613985,
2.631675659503586,
3.3278609055916517,
3.2532516423779314,
2.905534110063327
    ],
    "Premium AI": [
0.9887525201241133,
1.7978363566129574,
2.533273815037678,
2.2442033884399404,
2.6207111838069266
    ],
    "Random": [
0.15282961660861313,
0.23102673572767607,
-0.4133088595826076,
-1.0161904897531662,
0.266688324470925
    ],
    "Rule-based": [
-1.8113896021419555,
-1.6751793678449705,
-1.9418590311082542,
-0.7557639159018175,
-0.22399929085702852
    ],
    "Trained model": [
2.1204782748616378,
1.8189446355955616,
1.3985209206275007,
2.303926968562719,
2.2411241611208865
    ],
    "VADER": [
-0.29534447571794686,
-0.7759505286972372,
-1.7554428444589227,
-1.646612796519666,
-1.090488688143187
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
plt.ylabel("mean normalized score (%)")
plt.title("Mean normalized score po veličini izlaznog skupa")

plt.xticks(x, top_k)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("mean_normalized_po_k.png", dpi=300)
plt.show()