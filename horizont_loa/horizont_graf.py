import matplotlib.pyplot as plt

data = {
    "FinBERT": [
0.107334539375,
0.23214558421875
    ],
    "Mini": [
0.07774474839344263,
0.4149470383715847
    ],
    "Premium AI": [
0.05936496642622951,
0.30100603236065576
    ],
    "Random": [
0.0007321404166666605,
-0.02164953765625
    ],
    "Rule-based": [
0.0156608675,
-0.17551887723958332
    ],
    "Trained model": [
0.08738173208333333,
0.16494026859375002
    ],
    "VADER": [
-0.05480454916666667,
-0.16655610953124997
    ]
}


modes = ["1_DAY", "7_DAYS"]

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