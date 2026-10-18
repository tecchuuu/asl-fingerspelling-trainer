"""Check that normalize.py removes position, size and aspect, and plot raw vs normalized hands.

Run from the project root:  python -m scripts.check_normalize
"""

import matplotlib.pyplot as plt
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid

from src import config
from src.normalize import mirror, normalize

FIGURE_PATH = config.FIGURES_DIR / "normalize_before_after.png"
PLOT_LETTER = "A"
PLOT_SAMPLES = 40


def load_all():
    X, y = [], []
    for i, letter in enumerate(config.LETTERS):
        with np.load(config.PROCESSED_DIR / f"{letter}.npz") as data:
            X.append(data["landmarks"])
            y.append(np.full(len(data["landmarks"]), i))
    return np.concatenate(X), np.concatenate(y)


def invariance_checks(raw):
    base = normalize(raw)

    shifted = raw.copy()
    shifted[:, :2] += [0.2, -0.1]
    assert np.allclose(normalize(shifted), base, atol=1e-5), "position still matters"

    scaled = raw * 0.5
    assert np.allclose(normalize(scaled), base, atol=1e-5), "size still matters"

    # The same hand seen in a 640x480 frame: x and z shrink by 480/640
    squashed = raw.copy()
    squashed[:, [0, 2]] *= 480 / 640
    assert np.allclose(normalize(squashed, aspect=640 / 480), base, atol=1e-5), "aspect still matters"

    assert np.allclose(mirror(mirror(base)), base), "mirror twice should be a no-op"
    print("Invariance checks passed: position, size, aspect ratio, mirror")


def draw(ax, hands, title):
    for hand in hands:
        for conn in HandLandmarksConnections.HAND_CONNECTIONS:
            ax.plot(*zip(hand[conn.start, :2], hand[conn.end, :2]), color="tab:blue", alpha=0.15, linewidth=1)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # image y points down


def main():
    X_raw, y = load_all()
    X = normalize(X_raw)
    hands = X.reshape(-1, 21, 3)
    print(f"{len(X):,} samples -> features {X.shape}")

    invariance_checks(X_raw[0])

    palm = np.linalg.norm(hands[:, 9], axis=-1)
    print(f"Wrist max |value|: {np.abs(hands[:, 0]).max():.1e}   palm length: {palm.min():.3f}..{palm.max():.3f}")
    print(f"Feature range: x {hands[..., 0].min():+.2f}..{hands[..., 0].max():+.2f}  "
          f"y {hands[..., 1].min():+.2f}..{hands[..., 1].max():+.2f}  "
          f"z {hands[..., 2].min():+.2f}..{hands[..., 2].max():+.2f}")

    # Quick separability check, not the real model: classify by nearest letter-average
    for name, features in [("raw", X_raw.reshape(len(X_raw), -1)), ("normalized", X)]:
        X_tr, X_te, y_tr, y_te = train_test_split(features, y, test_size=0.2, random_state=config.SEED, stratify=y)
        acc = NearestCentroid().fit(X_tr, y_tr).score(X_te, y_te)
        print(f"Nearest-centroid accuracy, {name:>10}: {acc:.1%}")

    idx = np.flatnonzero(y == config.LETTERS.index(PLOT_LETTER))
    pick = np.random.default_rng(config.SEED).choice(idx, PLOT_SAMPLES, replace=False)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 4.5))
    draw(ax0, X_raw[pick], f"Raw: {PLOT_SAMPLES} '{PLOT_LETTER}' hands (image coordinates)")
    draw(ax1, hands[pick], "Normalized: wrist at 0, palm length 1")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=100)
    print(f"Figure: {FIGURE_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
