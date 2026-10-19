"""Final evaluation of the best MLP and CNN on the test set.

Picks each model's best run by VALIDATION F1 from runs.csv (the test set plays no part
in choosing), copies its weights to models/<model>_final.pt, then reports test scores,
per-letter F1, confusion matrices and inference speed.

Usage (from the project root):
    python -m src.evaluate
    python -m src.evaluate --mlp <run_id> --cnn <run_id>   # pick runs explicitly
"""

import argparse
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src import config
from src.dataset import crop_splits, landmark_splits, to_loader
from src.models import CNN, MLP, count_params

MODELS_DIR = config.ROOT / "models"
SUMMARY_CSV = config.RESULTS_DIR / "test_summary.csv"
PER_LETTER_CSV = config.RESULTS_DIR / "test_per_letter.csv"
SPEED_SAMPLES = 500


def best_run(model_name):
    with open(config.RUNS_CSV, newline="") as f:
        runs = [r for r in csv.DictReader(f) if r["model"] == model_name]
    return max(runs, key=lambda r: float(r["val_f1"]))["run_id"]


def load_model(run_id):
    ckpt = torch.load(config.CHECKPOINTS_DIR / f"{run_id}.pt", map_location="cpu")
    args = ckpt["args"]
    model = (MLP if args["model"] == "mlp" else CNN)(dropout=args["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(model, X, y):
    preds = [model(xb).argmax(1) for xb, _ in to_loader(X, y, batch_size=1024, shuffle=False)]
    return torch.cat(preds).numpy()


@torch.no_grad()
def ms_per_sample(model, X, y):
    """Median time for ONE sample, the way the live app calls the model (batch of 1)."""
    loader = to_loader(X[:SPEED_SAMPLES], y[:SPEED_SAMPLES], batch_size=1, shuffle=False)
    times = []
    for i, (xb, _) in enumerate(loader):
        t0 = time.perf_counter()
        model(xb)
        if i >= 20:  # skip warm-up calls
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def plot_confusions(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 7.5))
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        # Row-normalized: each row shows where that true letter's samples went
        cm = r["cm"] / r["cm"].sum(axis=1, keepdims=True)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(config.NUM_CLASSES), config.LETTERS)
        ax.set_yticks(range(config.NUM_CLASSES), config.LETTERS)
        ax.set(xlabel="predicted", ylabel="true", title=f"{name}: test macro F1 {r['f1']:.1%}")
        for i in range(config.NUM_CLASSES):
            for j in range(config.NUM_CLASSES):
                if i != j and cm[i, j] >= 0.02:
                    ax.text(j, i, f"{cm[i, j]:.0%}"[:-1], ha="center", va="center", fontsize=6, color="red")
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_per_letter(results, path):
    x = np.arange(config.NUM_CLASSES)
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(12, 4))
    for k, (name, r) in enumerate(results.items()):
        ax.bar(x + k * width - 0.4 + width / 2, r["per_letter"], width, label=name)
    ax.set_xticks(x, config.LETTERS)
    ax.set(ylim=(0.5, 1.0), ylabel="F1", title="Test F1 per letter")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def top_confusions(cm, n=8):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1][:n], off.shape))[0]
    return [(config.LETTERS[i], config.LETTERS[j], int(off[i, j]), off[i, j] / cm[i].sum()) for i, j in pairs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp", default=None, help="run_id (default: best val F1)")
    parser.add_argument("--cnn", default=None, help="run_id (default: best val F1)")
    args = parser.parse_args()

    run_ids = {"MLP": args.mlp or best_run("mlp"), "CNN": args.cnn or best_run("cnn")}
    test_sets = {
        "MLP": landmark_splits("block", mirror_train=False)["test"],
        "CNN": crop_splits("block", mirror_train=False)["test"],
    }
    assert np.array_equal(test_sets["MLP"][1], test_sets["CNN"][1]), "both models must see the same test samples"

    MODELS_DIR.mkdir(exist_ok=True)
    results = {}
    for name, run_id in run_ids.items():
        model, ckpt = load_model(run_id)
        final_path = MODELS_DIR / f"{name.lower()}_final.pt"
        torch.save({**ckpt, "run_id": run_id}, final_path)

        X, y = test_sets[name]
        preds = predict(model, X, y)
        results[name] = {
            "run_id": run_id,
            "acc": accuracy_score(y, preds),
            "f1": f1_score(y, preds, average="macro"),
            "per_letter": f1_score(y, preds, average=None),
            "cm": confusion_matrix(y, preds, labels=range(config.NUM_CLASSES)),
            "params": count_params(model),
            "size_kb": final_path.stat().st_size / 1024,
            "ms": ms_per_sample(model, X, y),
        }

    print(f"Test set: {len(test_sets['MLP'][1]):,} samples (block split)\n")
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "run_id", "test_acc", "test_macro_f1", "params", "file_kb", "ms_per_sample"])
        for name, r in results.items():
            writer.writerow([name, r["run_id"], f"{r['acc']:.4f}", f"{r['f1']:.4f}", r["params"],
                             f"{r['size_kb']:.0f}", f"{r['ms']:.3f}"])
            print(f"{name}: test acc {r['acc']:.2%}  macro F1 {r['f1']:.2%}  |  {r['params']:,} params, "
                  f"{r['size_kb']:.0f} KB, {r['ms']:.2f} ms/sample  ({r['run_id']})")

    with open(PER_LETTER_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["letter", *[f"{n}_f1" for n in results]])
        for i, letter in enumerate(config.LETTERS):
            writer.writerow([letter, *[f"{r['per_letter'][i]:.4f}" for r in results.values()]])

    for name, r in results.items():
        worst = np.argsort(r["per_letter"])[:5]
        print(f"\n{name} hardest letters: " + ", ".join(f"{config.LETTERS[i]} {r['per_letter'][i]:.1%}" for i in worst))
        print(f"{name} top confusions (true -> predicted):")
        for t, p, n, share in top_confusions(r["cm"]):
            print(f"  {t} -> {p}: {n} ({share:.1%} of {t})")

    plot_confusions(results, config.FIGURES_DIR / "confusion_matrices.png")
    plot_per_letter(results, config.FIGURES_DIR / "per_letter_f1.png")
    print(f"\nSaved: {SUMMARY_CSV.relative_to(config.ROOT)}, {PER_LETTER_CSV.relative_to(config.ROOT)}, "
          f"figures/confusion_matrices.png, figures/per_letter_f1.png, models/*_final.pt")


if __name__ == "__main__":
    main()
