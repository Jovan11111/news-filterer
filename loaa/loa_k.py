import matplotlib.pyplot as plt

top_k = [5, 10, 20, 50, 100]

data = {
    "FinBERT": [
0.3037466579285714,
0.22261182459523807,
0.1428736103095238,
0.09508057869444446,
0.0353147711
    ],
    "Mini": [
0.26422738805208335,
0.25374710680208334,
0.23015781513541667,
0.27165652658333334,
0.17674606476666668
    ],
    "Premium AI": [
0.15869442971875003,
0.19907081513541666,
0.18763888805208334,
0.19128381408333334,
0.14691576476666665
    ],
    "Random": [
0.028282467452380944,
0.016142776976190477,
-0.05185473492857142,
-0.05483452130555555,
0.009267041099999991
    ],
    "Rule-based": [
-0.19374510397619046,
-0.10686729445238094,
-0.0536051992142857,
-0.019552299083333325,
0.007821764433333338
    ],
    "Trained model": [
0.23713118173809525,
0.1281345388809524,
0.07551397935714288,
0.08012062869444446,
0.09419406776666668
    ],
    "VADER": [
-0.1568853420714286,
-0.1285154373095238,
-0.11308463969047619,
-0.07511154352777777,
-0.06034066889999999
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
plt.ylabel("Vrednost poboljšanja u odnosu na prosek (%)")
plt.title("Vrednost poboljšanja u odnosu na prosek po veličini izlaznog skupa")

plt.xticks(x, top_k)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("vrednost_poboljsanja_prosek_po_k.png", dpi=300)
plt.show()