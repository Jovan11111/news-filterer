import matplotlib.pyplot as plt

data = {
    "FinBERT": [
0.162627466640625,
0.081940211484375,
0.264652507265625
    ],
    "Mini": [
0.13639552424590165,
-0.12435197752459014,
0.7269941334262295
    ],
    "Premium AI": [
0.2715607488360656,
-0.23143693818032787,
0.5004326875245902
    ],
    "Random": [
-0.07267226773437499,
0.023901328671874986,
0.017394843203125003
    ],
    "Rule-based": [
0.143838904140625,
-0.375139074453125,
-0.00848684429687499
    ],
    "Trained model": [
0.15774493539062504,
0.11253379117187498,
0.10820427445312503
    ],
    "VADER": [
-0.065073539609375,
-0.11936155882812499,
-0.14760588960937499
    ]
}


modes = ["BUY", "SELL", "AFFECT"]

for mode_index, mode in enumerate(modes):

    filters = list(data.keys())
    values = [
        data[filter_name][mode_index]
        for filter_name in filters
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(filters, values)

    plt.axhline(0, linewidth=1)

    plt.xlabel("Filter")
    plt.ylabel("Vrednost poboljšanja u odnosu na prosek")
    plt.title(f"Vrednost poboljšanja u odnosu na prosek, {mode} režim")

    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"lift_over_average_{mode.lower()}.png",
        dpi=300
    )

    plt.show()