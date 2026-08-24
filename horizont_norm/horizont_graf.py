import matplotlib.pyplot as plt

data = {
    "FinBERT": [
1.1004026530515567,
1.6266444427267053
    ],
    "Mini": [
1.8159883766465692,
3.565129194417243
    ],
    "Premium AI": [
1.3331910989536335,
2.4758248385857153
    ],
    "Random": [
0.07155831321813723,
-0.3821751173986749
    ],
    "Rule-based": [
-0.7079654311204648,
-2.0203830662146793
    ],
    "Trained model": [
2.9140112811272503,
0.9856630585336028
    ],
    "VADER": [
-0.25901619906292855,
-1.9359391235591137
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
    plt.ylabel("Normalizovan rezultat filtera")
    plt.title(f"Normalizovan rezultat filtera, {mode} režim")

    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"normalized_score_{mode.lower()}.png",
        dpi=300
    )

    plt.show()