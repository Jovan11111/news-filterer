import matplotlib.pyplot as plt

top_k = [5, 10, 20, 50, 100]

data = {
    "FinBERT": [
        0.11396323311127303,
        0.10196717031150551,
        0.09201246610830052,
        0.06794240160890815,
        0.046347664801126155
    ],
    "Mini": [
        0.16833427979308532,
        0.13591553412397595,
        0.11764711725794105,
        0.06320871948193223,
        0.056013590596702176
    ],
    "Premium AI": [
        0.1642646483852396,
        0.1316668000613007,
        0.10616216304655476,
        0.05175392080830606,
        0.04587059989747354
    ],
    "Random": [
        0.13342955952395721,
        0.09939847573349873,
        0.10237278527569868,
        0.0720387355701138,
        0.05690573426483807
    ],
    "Rule-based": [
        0.12391290839012754,
        0.10070251245109164,
        0.10132054593859045,
        0.07751321937105699,
        0.05186284634639302
    ],
    "Trained model": [
        0.1174919642710578,
        0.1092081876762849,
        0.09972746255424432,
        0.07551218613373688,
        0.0584147781599715
    ],
    "VADER": [
        0.1077013470059701,
        0.09624394836962849,
        0.08499738590529578,
        0.06451154553149398,
        0.04716714497181017
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
plt.ylabel("Standardna devijacija")
plt.title("Konzistentnost filtera u zavisnosti od veličine izlaznog skupa")

plt.xticks(x, top_k)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("konzistentnost_po_top_k.png", dpi=300)
plt.show()