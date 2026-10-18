"""Plot validation curves of several runs on one figure.

Run from the project root:
    python -m scripts.plot_curves --notes "optimizer comparison" "first MLP" --out optimizer_comparison.png
Runs are picked from results/runs.csv by a substring of their notes.
"""

import argparse
import csv

import matplotlib.pyplot as plt

from src import config


def load_runs(note_filters):
    with open(config.RUNS_CSV, newline="") as f:
        return [r for r in csv.DictReader(f) if any(n in r["notes"] for n in note_filters)]


def load_history(run_id):
    with open(config.HISTORY_DIR / f"{run_id}.csv", newline="") as f:
        return list(csv.DictReader(f))


def label(run):
    name = "Adam" if run["optimizer"] == "adam" else "SGD+momentum"
    return f"{name} lr={run['lr']} bs={run['batch_size']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="file name inside results/figures/")
    args = parser.parse_args()

    runs = load_runs(args.notes)
    if not runs:
        parser.error("no runs match those notes")

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4.5))
    for run in runs:
        hist = load_history(run["run_id"])
        epochs = [int(h["epoch"]) for h in hist]
        line, = ax_loss.plot(epochs, [float(h["val_loss"]) for h in hist], label=label(run))
        ax_f1.plot(epochs, [float(h["val_f1"]) for h in hist], color=line.get_color(), label=label(run))
        best = int(run["best_epoch"])
        ax_f1.scatter([best], [float(run["val_f1"])], color=line.get_color(), zorder=3)

    ax_loss.set(title="Validation loss", xlabel="epoch", ylabel="cross-entropy")
    ax_f1.set(title="Validation macro F1 (dot = best epoch)", xlabel="epoch", ylabel="macro F1")
    ax_f1.legend(fontsize=8)
    for ax in (ax_loss, ax_f1):
        ax.grid(alpha=0.3)
    fig.tight_layout()

    out = config.FIGURES_DIR / args.out
    fig.savefig(out, dpi=100)
    print(f"{len(runs)} runs plotted -> {out.relative_to(config.ROOT)}")
    for run in runs:
        print(f"  {label(run):<32} val F1 {float(run['val_f1']):.3f}  best epoch {run['best_epoch']:>2}/"
              f"{run['epochs_run']:>2}  {run['train_seconds']:>4} s")


if __name__ == "__main__":
    main()
