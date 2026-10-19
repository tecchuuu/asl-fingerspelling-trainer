"""Compare the 24-class MLP with the 25-class MLP ("other" class for non-letter hands).

Uses the latest mlp_other run (or --run), saves it as models/mlp_other_final.pt, then reports:
  - letters: macro F1 over the 24 letters, and how often a real letter is wrongly called "other"
  - non-letter hands (held-out other-test session): how often each model would type a letter

Run from the project root:  python -m scripts.compare_other
"""

import argparse
import csv
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score

from src import config
from src.dataset import OTHER_TEST_SESSIONS, landmark_splits, load_other, pinch
from src.evaluate_webcam import MODELS_DIR, STANDARD_PREFIX, load_final, load_session
from src.predict import PINCH_GUARD

OUT_CSV = config.RESULTS_DIR / "other_class_comparison.csv"
TYPE_CONF = 0.8  # the gate's confidence threshold


@torch.no_grad()
def probs_of(model, X, guard=False):
    probs = torch.softmax(model(torch.from_numpy(X).float()), dim=1).numpy()
    if guard and probs.shape[1] > config.NUM_CLASSES:
        pinched = pinch(X) < PINCH_GUARD
        probs[pinched, config.OTHER_INDEX] = 0
        probs[pinched] /= probs[pinched].sum(axis=1, keepdims=True)
    return probs


def letter_scores(probs, y):
    pred = probs.argmax(1)
    return {
        "f1": f1_score(y, pred, average="macro", labels=range(config.NUM_CLASSES)),
        "called_other": float(np.mean(pred == config.OTHER_INDEX)),
    }


def non_letter_scores(probs):
    """For hands that are NOT letters: share recognized as other, and share confident enough to be typed."""
    pred, conf = probs.argmax(1), probs.max(1)
    is_letter = pred < config.NUM_CLASSES
    typed = is_letter & (conf >= TYPE_CONF)
    common = Counter(config.LETTERS[p] for p in pred[typed]).most_common(3)
    return {
        "recognized_other": float(np.mean(~is_letter)),
        "would_type": float(np.mean(typed)),
        "typed_as": ", ".join(f"{l} {n}" for l, n in common) or "-",
    }


def latest_run(model_name):
    with open(config.RUNS_CSV, newline="") as f:
        return [r["run_id"] for r in csv.DictReader(f) if r["model"] == model_name][-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="mlp_other run_id (default: the latest)")
    args = parser.parse_args()
    # Latest, not best val F1: runs with different "other" cleaning have different val sets
    run_id = args.run or latest_run("mlp_other")
    ckpt = torch.load(config.CHECKPOINTS_DIR / f"{run_id}.pt", map_location="cpu")
    torch.save({**ckpt, "run_id": run_id}, MODELS_DIR / "mlp_other_final.pt")
    other_model = load_final("mlp_other")
    models = {"MLP (24)": (load_final("mlp"), False), "MLP + other (25)": (other_model, False),
              "+ pinch guard": (other_model, True)}
    print(f"MLP + other: run {run_id}, saved to models/mlp_other_final.pt\n")

    test_X, test_y = landmark_splits("block", mirror_train=False)["test"]
    sessions = [p.stem for p in config.WEBCAM_TEST_DIR.glob("*.npz") if not p.stem.startswith(STANDARD_PREFIX)]
    webcam = [load_session(s) for s in sorted(sessions)]
    web_X = np.concatenate([f for f, _, y in webcam if len(y)])
    web_y = np.concatenate([y for _, _, y in webcam if len(y)])
    # Uncleaned: real non-letter hands include the F-like ones too
    other_X = load_other(OTHER_TEST_SESSIONS, clean=False)

    rows = []
    for name, (model, guard) in models.items():
        d = letter_scores(probs_of(model, test_X, guard), test_y)
        w = letter_scores(probs_of(model, web_X, guard), web_y)
        o = non_letter_scores(probs_of(model, other_X, guard))
        rows.append([name, d["f1"], d["called_other"], w["f1"], w["called_other"],
                     o["recognized_other"], o["would_type"], o["typed_as"]])

    header = ["model", "dataset_letter_f1", "dataset_letters_called_other", "webcam_letter_f1",
              "webcam_letters_called_other", "nonletter_recognized_other", "nonletter_would_type", "typed_as"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows([[r[0], *[f"{v:.4f}" for v in r[1:7]], r[7]] for r in rows])

    print(f"Letters: dataset test {len(test_y):,} samples, webcam {len(web_y)} samples (3 sessions)")
    print(f"Non-letter hands: {len(other_X)} samples from held-out session(s) {', '.join(OTHER_TEST_SESSIONS)}\n")
    print(f"{'':<18}{'dataset F1':>11}{'->other':>9}{'webcam F1':>11}{'->other':>9}"
          f"{'| non-letter: is other':>23}{'would type':>12}  typed as")
    for r in rows:
        print(f"{r[0]:<18}{r[1]:>11.1%}{r[2]:>9.1%}{r[3]:>11.1%}{r[4]:>9.1%}{r[5]:>23.1%}{r[6]:>12.1%}  {r[7]}")
    print(f"\n'would type' = predicted a letter with confidence >= {TYPE_CONF} (enough for the gate). Saved {OUT_CSV.name}")


if __name__ == "__main__":
    main()
