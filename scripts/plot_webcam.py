"""Bar chart: dataset test F1 vs each webcam session, MLP vs CNN.

Run from the project root (after src.evaluate and src.evaluate_webcam):  python -m scripts.plot_webcam
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from src import config

OUT = config.FIGURES_DIR / "webcam_vs_dataset.png"


def main():
    with open(config.RESULTS_DIR / "test_summary.csv", newline="") as f:
        test = {r["model"]: float(r["test_macro_f1"]) for r in csv.DictReader(f)}
    with open(config.RESULTS_DIR / "webcam_summary.csv", newline="") as f:
        webcam = list(csv.DictReader(f))

    groups = ["Dataset test"] + [r["session"] if r["session"] != "ALL" else "All webcam" for r in webcam]
    mlp = [test["MLP"]] + [float(r["mlp_f1"]) for r in webcam]
    cnn = [test["CNN"]] + [float(r["cnn_f1"]) for r in webcam]

    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for offset, values, name in [(-0.2, mlp, "MLP (landmarks)"), (0.2, cnn, "CNN (pixels)")]:
        bars = ax.bar(x + offset, values, 0.4, label=name)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in values], fontsize=8, padding=2)
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=1)
    ax.text(len(groups) - 0.5, 0.81, "80% webcam target", ha="right", fontsize=8, color="grey")
    ax.axvline(0.5, color="black", linewidth=0.8)
    ax.set_xticks(x, groups)
    ax.set(ylim=(0, 1.08), ylabel="macro F1", title="Dataset test vs real webcam")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=100)
    print(f"Saved {OUT.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
