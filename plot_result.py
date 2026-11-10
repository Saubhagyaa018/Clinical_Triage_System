import matplotlib.pyplot as plt
import numpy as np

methods = [
    "RF\nVitals",
    "RF\nVitals + Complaint",
    "NEWS2"
]

scores = [
    0.438,
    0.440,
    0.281
]

errors = [
    0.054,
    0.068,
    0.0
]

x = np.arange(len(methods))

plt.figure(figsize=(8, 5))

plt.bar(
    x,
    scores,
    yerr=errors,
    capsize=5
)

plt.xticks(x, methods)
plt.ylabel("Macro-F1")
plt.xlabel("Method")
plt.title("Macro-F1 Comparison on MIMIC-IV-ED")
plt.ylim(0, 0.55)

for i, score in enumerate(scores):
    plt.text(
        i,
        score + 0.015,
        f"{score:.3f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig("macro_f1_comparison.png", dpi=300)
plt.show()