import matplotlib.pyplot as plt

data = {
    "FinBERT": [
0.169740061796875

    ],
    "Mini": [
0.24634589338251367

    ],
    "Premium AI": [
0.18018549939344264

    ],
    "Random": [
-0.010458698619791668

    ],
    "Rule-based": [
-0.07992900486979165

    ],
    "Trained model": [
0.1261610003385417

    ],
    "VADER": [
-0.11068032934895833
    ]
}


filters = list(data.keys())
values = [
    data[filter_name][0]
    for filter_name in filters
]

plt.figure(figsize=(10, 6))

plt.bar(filters, values)

plt.axhline(0, linewidth=1)

plt.xlabel("Filter")
plt.ylabel("Vrednost poboljšanja u odnosu na prosek")
plt.title(f"Vrednost poboljšanja u odnosu na prosek")

plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"lift_over_average.png",
    dpi=300
)

plt.show()