import matplotlib.pyplot as plt

data = {
    "FinBERT": [
1.3635235478891312
    ],
    "Mini": [
2.6905587855319055
    ],
    "Premium AI": [
1.9045079687696742
    ],
    "Random": [
-0.15530840209026878
    ],
    "Rule-based": [
-1.3641742486675723
    ],
    "Trained model": [
1.9498371698304267
    ],
    "VADER": [
-1.0974776613110213
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
plt.ylabel("Normalizovan rezultat filtera")
plt.title(f"Normalizovan rezultat filtera")

plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"normalized_results_.png",
    dpi=300
)

plt.show()