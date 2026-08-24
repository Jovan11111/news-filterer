import matplotlib.pyplot as plt

data = {
    "FinBERT": [
1.346344341781944,
-0.13224664544544343,
2.8764729473308925
    ],
    "Mini": [
1.0781090912408735,
-1.842539026496096,
8.83610629185094
    ],
    "Premium AI": [
2.1234351938352614,
-3.0039917705173185,
6.594080482991081
    ],
    "Random": [
-1.0183790954026881,
0.07743633044326016,
0.47501755868862167
    ],
    "Rule-based": [
0.8243049098313807,
-5.365881134202633,
0.4490534783685351
    ],
    "Trained model": [
1.968375892632854,
1.168049687569395,
2.7130859292890306
    ],
    "VADER": [
-0.7625390510683238,
-2.115421767329889,
-0.4144721655348509
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
    plt.ylabel("Normalizovan rezultat filtera")
    plt.title(f"Normalizovan rezultat filtera, {mode} režim")

    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"normalized_results_{mode.lower()}.png",
        dpi=300
    )

    plt.show()