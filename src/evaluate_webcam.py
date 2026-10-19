"""Score the final MLP and CNN on the recorded webcam test sessions.

By default prints only overall scores per session, so recording later sessions
isn't biased by knowing which letters fail. Use --details once all sessions are in.

Usage (from the project root):
    python -m src.evaluate_webcam                        # all sessions, overall only
    python -m src.evaluate_webcam --sessions room-right  # some sessions
    python -m src.evaluate_webcam --details              # + per-letter F1 and confusions
"""

import argparse
import csv

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from src import config
from src.evaluate import top_confusions
from src.models import CNN, MLP
from src.normalize import normalize

MODELS_DIR = config.ROOT / "models"
SUMMARY_CSV = config.RESULTS_DIR / "webcam_summary.csv"
# Sessions signed in standard ASL where the dataset differs (T, K, P). Kept out of the
# main webcam score: they measure the dataset's labelling, not real-world robustness.
STANDARD_PREFIX = "standard"


def load_final(name):
    """name: "mlp", "mlp_other" or "cnn" -> models/<name>_final.pt"""
    ckpt = torch.load(MODELS_DIR / f"{name}_final.pt", map_location="cpu")
    if name == "cnn":
        model = CNN(dropout=ckpt["args"]["dropout"])
    else:
        num_classes = config.NUM_CLASSES + (ckpt["args"]["model"] == "mlp_other")
        model = MLP(dropout=ckpt["args"]["dropout"], num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_session(name):
    with np.load(config.WEBCAM_TEST_DIR / f"{name}.npz") as d:
        labels = np.array([config.LETTERS.index(str(l)) for l in d["labels"]], dtype=int)
        if len(labels) == 0:
            return None, None, labels
        aspect = d["frame_size"][:, 0] / d["frame_size"][:, 1]
        features = np.stack([normalize(lm, aspect=a) for lm, a in zip(d["landmarks"], aspect)])
        crops = d["crops"]
    return features, crops, labels


@torch.no_grad()
def predict(models, features, crops):
    x_crops = torch.from_numpy(crops).permute(0, 3, 1, 2).float() / 255
    return {
        "MLP": models["mlp"](torch.from_numpy(features).float()).argmax(1).numpy(),
        "CNN": models["cnn"](x_crops).argmax(1).numpy(),
    }


def report_standard_asl(session, models):
    """Per letter: how often each model accepts the standard-ASL version, and what it predicts instead."""
    features, crops, labels = load_session(session)
    if len(labels) == 0:
        return
    preds = predict(models, features, crops)
    print(f"{session} (standard ASL, not in the main score):")
    for letter_idx in np.unique(labels):
        mask = labels == letter_idx
        parts = []
        for m, p in preds.items():
            got = p[mask]
            others = [config.LETTERS[i] for i in got[got != letter_idx]]
            common = max(set(others), key=others.count) if others else "-"
            parts.append(f"{m} accepts {np.mean(got == letter_idx):.0%} (else mostly {common})")
        print(f"  {config.LETTERS[letter_idx]} (n={mask.sum()}): " + " | ".join(parts))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", nargs="+", default=None, help="default: every session in data/webcam_test")
    parser.add_argument("--details", action="store_true", help="also show per-letter F1 and top confusions")
    args = parser.parse_args()

    sessions = args.sessions or sorted(p.stem for p in config.WEBCAM_TEST_DIR.glob("*.npz"))
    models = {"mlp": load_final("mlp"), "cnn": load_final("cnn")}
    for s in [s for s in sessions if s.startswith(STANDARD_PREFIX)]:
        report_standard_asl(s, models)
    sessions = [s for s in sessions if not s.startswith(STANDARD_PREFIX)]

    rows, all_labels, all_preds = [], [], {"MLP": [], "CNN": []}
    print(f"{'session':<16}{'n':>5}{'MLP F1':>10}{'CNN F1':>10}{'MLP acc':>10}{'CNN acc':>10}")
    for s in sessions:
        features, crops, labels = load_session(s)
        if len(labels) == 0:
            print(f"{s:<16}    0  (empty, skipped)")
            continue
        preds = predict(models, features, crops)
        f1 = {m: f1_score(labels, p, average="macro") for m, p in preds.items()}
        acc = {m: accuracy_score(labels, p) for m, p in preds.items()}
        print(f"{s:<16}{len(labels):>5}{f1['MLP']:>10.1%}{f1['CNN']:>10.1%}{acc['MLP']:>10.1%}{acc['CNN']:>10.1%}")
        rows.append([s, len(labels), f"{f1['MLP']:.4f}", f"{f1['CNN']:.4f}", f"{acc['MLP']:.4f}", f"{acc['CNN']:.4f}"])
        all_labels.append(labels)
        for m in preds:
            all_preds[m].append(preds[m])

    y = np.concatenate(all_labels)
    pooled = {m: np.concatenate(p) for m, p in all_preds.items()}
    f1 = {m: f1_score(y, p, average="macro") for m, p in pooled.items()}
    acc = {m: accuracy_score(y, p) for m, p in pooled.items()}
    print(f"{'ALL':<16}{len(y):>5}{f1['MLP']:>10.1%}{f1['CNN']:>10.1%}{acc['MLP']:>10.1%}{acc['CNN']:>10.1%}")
    rows.append(["ALL", len(y), f"{f1['MLP']:.4f}", f"{f1['CNN']:.4f}", f"{acc['MLP']:.4f}", f"{acc['CNN']:.4f}"])

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session", "n", "mlp_f1", "cnn_f1", "mlp_acc", "cnn_acc"])
        writer.writerows(rows)

    if args.details:
        for m, p in pooled.items():
            per_letter = f1_score(y, p, average=None, labels=range(config.NUM_CLASSES))
            worst = np.argsort(per_letter)[:6]
            print(f"\n{m} hardest letters: " + ", ".join(f"{config.LETTERS[i]} {per_letter[i]:.0%}" for i in worst))
            cm = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=int)
            np.add.at(cm, (y, p), 1)
            for t, pr, n, share in top_confusions(cm, n=6):
                print(f"  {t} -> {pr}: {n} ({share:.0%} of {t})")


if __name__ == "__main__":
    main()
