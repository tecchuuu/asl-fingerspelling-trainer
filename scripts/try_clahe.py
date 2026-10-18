"""Retry the images MediaPipe failed on, after brightening, and count how many are recovered.

Run from the project root:  python -m scripts.try_clahe
"""

import csv
import os
from multiprocessing import Pool

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections

from src import config, extract

OUT_CSV = config.RESULTS_DIR / "clahe_experiment.csv"
GRID_PATH = config.FIGURES_DIR / "gamma_recovered.png"


def clahe(rgb):
    """Equalize contrast locally on the lightness channel only, so colors don't shift."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def gamma(rgb, g=0.5):
    table = ((np.arange(256) / 255.0) ** g * 255).astype(np.uint8)
    return cv2.LUT(rgb, table)


METHODS = {
    "clahe": clahe,
    "gamma": gamma,
    "clahe+gamma": lambda rgb: clahe(gamma(rgb)),
}


def retry_letter(letter):
    with np.load(config.PROCESSED_DIR / f"{letter}.npz") as data:
        failed = list(data["failed"])

    recovered = {name: 0 for name in METHODS}
    any_method = 0
    examples = []
    for name in failed:
        bgr = cv2.imread(str(config.RAW_TRAIN_DIR / letter / name))
        t = extract.BORDER_TRIM
        rgb = cv2.cvtColor(bgr[t:-t, t:-t], cv2.COLOR_BGR2RGB)
        hit = False
        for method, fn in METHODS.items():
            enhanced = fn(rgb)
            result = extract._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced))
            if result.hand_landmarks:
                recovered[method] += 1
                if method == "gamma" and len(examples) < 2:
                    examples.append((rgb, enhanced, result.hand_landmarks[0], f"{letter}/{name}"))
                hit = True
        any_method += hit
    return letter, len(failed), recovered, any_method, examples


def save_grid(examples, path):
    rows = len(examples)
    fig, axes = plt.subplots(rows, 2, figsize=(5, 2.5 * rows))
    for (orig, enhanced, hand, title), (ax0, ax1) in zip(examples, axes):
        ax0.imshow(orig)
        ax0.set_title(f"{title} original", fontsize=7)
        ax1.imshow(enhanced)
        ax1.set_title("gamma 0.5 + landmarks", fontsize=7)
        h, w = enhanced.shape[:2]
        pts = [(lm.x * w, lm.y * h) for lm in hand]
        for conn in HandLandmarksConnections.HAND_CONNECTIONS:
            (x1, y1), (x2, y2) = pts[conn.start], pts[conn.end]
            ax1.plot([x1, x2], [y1, y2], color="lime", linewidth=1)
        ax0.axis("off")
        ax1.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    workers = max(1, os.cpu_count() - 2)
    rows, all_examples = [], []
    with Pool(workers, initializer=extract._init_worker) as pool:
        for letter, n_failed, recovered, any_method, examples in pool.imap(retry_letter, config.LETTERS):
            rates = "  ".join(f"{m}={recovered[m] / n_failed:.0%}" for m in METHODS)
            print(f"  {letter}: {n_failed} failed -> {rates}  any={any_method / n_failed:.0%}")
            rows.append([letter, n_failed, *recovered.values(), any_method])
            all_examples += examples

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["letter", "failed", *METHODS, "any"])
        writer.writerows(rows)

    total_failed = sum(r[1] for r in rows)
    print(f"\nTotal failed: {total_failed}")
    for i, method in enumerate([*METHODS, "any"], start=2):
        n = sum(r[i] for r in rows)
        print(f"  {method}: recovered {n} ({n / total_failed:.1%})")

    rng = np.random.default_rng(config.SEED)
    picks = rng.choice(len(all_examples), size=min(8, len(all_examples)), replace=False)
    save_grid([all_examples[i] for i in picks], GRID_PATH)
    print(f"Grid: {GRID_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
