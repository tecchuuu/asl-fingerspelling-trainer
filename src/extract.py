"""Run MediaPipe over the dataset and save landmarks + hand crops per letter.

Usage (from the project root):
    python -m src.extract                    # all letters, skips ones already done
    python -m src.extract --letters A B      # only some letters
    python -m src.extract --limit 50         # quick test, nothing is saved
"""

import argparse
import csv
import os
import time
from multiprocessing import Pool

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from src import config

SUMMARY_CSV = config.RESULTS_DIR / "extraction_summary.csv"
# The dataset photos have a ~2 px blue frame. Webcam frames don't, so the CNN must not see it.
BORDER_TRIM = 3
GAMMA = 0.5

_landmarker = None


def _init_worker():
    global _landmarker
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.HAND_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
    )
    _landmarker = HandLandmarker.create_from_options(options)


def crop_hand(rgb, landmarks):
    """Square crop around the landmarks, black-padded so the hand stays centered at the image edge."""
    h, w = rgb.shape[:2]
    xs = np.array([lm.x for lm in landmarks]) * w
    ys = np.array([lm.y for lm in landmarks]) * h
    side = int(max(np.ptp(xs), np.ptp(ys)) * (1 + 2 * config.CROP_MARGIN))
    cx, cy = int(xs.mean()), int(ys.mean())

    padded = cv2.copyMakeBorder(rgb, side, side, side, side, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    x0, y0 = cx - side // 2 + side, cy - side // 2 + side
    crop = padded[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (config.CROP_SIZE, config.CROP_SIZE), interpolation=cv2.INTER_AREA)


def detect(rgb):
    return _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))


def brighten(rgb):
    """Gamma 0.5 lifts dark (backlit) hands. Recovered 43% of failures; CLAHE recovered 2%."""
    table = ((np.arange(256) / 255.0) ** GAMMA * 255).astype(np.uint8)
    return cv2.LUT(rgb, table)


def process_letter(args):
    letter, limit = args
    paths = sorted((config.RAW_TRAIN_DIR / letter).glob("*.jpg"))[:limit]

    landmarks, crops, files, handedness, scores, enhanced, failed = [], [], [], [], [], [], []
    start = time.perf_counter()
    for path in paths:
        bgr = cv2.imread(str(path))
        rgb = cv2.cvtColor(bgr[BORDER_TRIM:-BORDER_TRIM, BORDER_TRIM:-BORDER_TRIM], cv2.COLOR_BGR2RGB)
        result = detect(rgb)
        was_enhanced = False
        if not result.hand_landmarks:
            result = detect(brighten(rgb))
            was_enhanced = True
        if not result.hand_landmarks:
            failed.append(path.name)
            continue
        hand = result.hand_landmarks[0]
        landmarks.append([(lm.x, lm.y, lm.z) for lm in hand])
        # Crop the original pixels, not the brightened ones: the CNN should see real images
        crops.append(crop_hand(rgb, hand))
        files.append(path.name)
        handedness.append(result.handedness[0][0].category_name)
        scores.append(result.handedness[0][0].score)
        enhanced.append(was_enhanced)

    return {
        "letter": letter,
        "total": len(paths),
        "seconds": time.perf_counter() - start,
        "landmarks": np.array(landmarks, dtype=np.float32).reshape(-1, config.NUM_LANDMARKS, 3),
        "crops": np.array(crops, dtype=np.uint8).reshape(-1, config.CROP_SIZE, config.CROP_SIZE, 3),
        "files": np.array(files, dtype=str),
        "handedness": np.array(handedness, dtype=str),
        "handedness_score": np.array(scores, dtype=np.float32),
        "enhanced": np.array(enhanced, dtype=bool),
        "failed": np.array(failed, dtype=str),
    }


def is_done(letter):
    out = config.PROCESSED_DIR / f"{letter}.npz"
    if not out.exists():
        return False
    with np.load(out) as data:
        return int(data["total"]) == len(list((config.RAW_TRAIN_DIR / letter).glob("*.jpg")))


def write_summary():
    rows = []
    for letter in config.LETTERS:
        out = config.PROCESSED_DIR / f"{letter}.npz"
        if not out.exists():
            continue
        with np.load(out) as data:
            total, found = int(data["total"]), len(data["files"])
            n_enhanced = int(data["enhanced"].sum())
        rows.append([letter, total, found, found - n_enhanced, n_enhanced, total - found,
                     f"{(total - found) / total:.3f}"])

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["letter", "total", "found", "found_original", "found_enhanced", "failed", "fail_rate"])
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--letters", nargs="+", default=config.LETTERS)
    parser.add_argument("--limit", type=int, default=None, help="images per letter, for testing (not saved)")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument("--force", action="store_true", help="redo letters that are already extracted")
    args = parser.parse_args()

    unknown = set(args.letters) - set(config.LETTERS)
    if unknown:
        parser.error(f"unknown letters: {sorted(unknown)}")

    todo = [l for l in args.letters if args.force or args.limit or not is_done(l)]
    skipped = len(args.letters) - len(todo)
    if skipped:
        print(f"Skipping {skipped} letters already extracted (use --force to redo)")
    if not todo:
        write_summary()
        return

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {len(todo)} letters with {args.workers} workers...")

    start = time.perf_counter()
    with Pool(args.workers, initializer=_init_worker) as pool:
        for res in pool.imap_unordered(process_letter, [(l, args.limit) for l in todo]):
            found = len(res["files"])
            fail_rate = 1 - found / res["total"]
            print(f"  {res['letter']}: {found}/{res['total']} found ({res['enhanced'].sum()} after brightening, "
                  f"{fail_rate:.1%} failed) in {res['seconds']:.0f} s")
            if args.limit is None:
                np.savez(config.PROCESSED_DIR / f"{res['letter']}.npz", total=res["total"],
                         **{k: v for k, v in res.items() if k not in ("letter", "total", "seconds")})

    print(f"Done in {(time.perf_counter() - start) / 60:.1f} min")
    if args.limit is None:
        rows = write_summary()
        found = sum(r[2] for r in rows)
        total = sum(r[1] for r in rows)
        print(f"Total: {found}/{total} found ({1 - found / total:.1%} failed). Summary: {SUMMARY_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
