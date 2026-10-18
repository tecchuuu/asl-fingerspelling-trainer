"""Run MediaPipe on one letter's images and check the landmarks look sane."""

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
TRAIN_DIR = ROOT / "data" / "raw" / "asl_alphabet_train"
FIGURES_DIR = ROOT / "results" / "figures"
KEPT_IMAGES = 75_000
GRID_SIZE = 9


def detect(landmarker, path):
    bgr = cv2.imread(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return rgb, result


def save_grid(samples, letter, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, (rgb, result, name) in zip(axes.flat, samples):
        ax.imshow(rgb)
        ax.set_title(name, fontsize=8)
        ax.axis("off")
        if not result.hand_landmarks:
            ax.set_title(f"{name}\nNO HAND", fontsize=8, color="red")
            continue
        h, w = rgb.shape[:2]
        pts = [(lm.x * w, lm.y * h) for lm in result.hand_landmarks[0]]
        for conn in HandLandmarksConnections.HAND_CONNECTIONS:
            (x1, y1), (x2, y2) = pts[conn.start], pts[conn.end]
            ax.plot([x1, x2], [y1, y2], color="lime", linewidth=1.5)
        ax.scatter(*zip(*pts), s=8, color="red")
    fig.suptitle(f"Letter {letter}: MediaPipe landmarks")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("letter", nargs="?", default="A")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N images")
    args = parser.parse_args()

    letter_dir = TRAIN_DIR / args.letter
    paths = sorted(letter_dir.glob("*.jpg"))[: args.limit]
    if not paths:
        print(f"No images found in {letter_dir}")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
    )

    rng = np.random.default_rng(0)
    grid_idx = set(rng.choice(len(paths), size=min(GRID_SIZE, len(paths)), replace=False).tolist())

    landmarks = []
    failed = []
    handedness = Counter()
    grid_samples = []

    start = time.perf_counter()
    with HandLandmarker.create_from_options(options) as landmarker:
        for i, path in enumerate(paths):
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(paths)}")
            rgb, result = detect(landmarker, path)
            if i in grid_idx:
                grid_samples.append((rgb, result, path.name))
            if not result.hand_landmarks:
                failed.append(path.name)
                continue
            handedness[result.handedness[0][0].category_name] += 1
            landmarks.append([(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]])
    elapsed = time.perf_counter() - start

    ms_per_image = elapsed * 1000 / len(paths)
    print(f"\nLetter {args.letter}: {len(paths)} images in {elapsed:.1f} s ({ms_per_image:.1f} ms/image)")
    print(f"Estimated time for all {KEPT_IMAGES:,} images: {KEPT_IMAGES * ms_per_image / 60000:.0f} min")
    print(f"Hand found: {len(landmarks)}   failed: {len(failed)} ({len(failed) / len(paths):.1%})")
    if failed:
        print(f"  first failures: {', '.join(failed[:5])}")
    print(f"Handedness: {dict(handedness)}")

    if landmarks:
        arr = np.array(landmarks)  # (n, 21, 3)
        print(f"Array shape: {arr.shape}")
        for axis, name in enumerate("xyz"):
            vals = arr[:, :, axis]
            print(f"  {name}: min={vals.min():+.3f}  max={vals.max():+.3f}  mean={vals.mean():+.3f}")
        wrist_z = np.abs(arr[:, 0, 2]).max()
        print(f"  wrist z max |value|: {wrist_z:.3f} (should be ~0, z is relative to the wrist)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"check_{args.letter}.png"
    save_grid(grid_samples, args.letter, out_path)
    print(f"Sample grid saved to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
