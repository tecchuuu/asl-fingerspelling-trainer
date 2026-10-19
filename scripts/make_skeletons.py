"""Export one typical hand skeleton per letter for the web app's drawings (logo, home hero).

For each letter: the 2-D normalized landmarks of the training sample closest to the letter's
median shape, mirrored to match the app's mirrored camera view. Written to
web/src/lib/skeletons.json as {letter: [[x, y], ... 21 points]} in palm lengths, wrist at (0, 0).

Run from the project root:  python -m scripts.make_skeletons
"""

import json

import numpy as np

from src import config
from src.normalize import normalize

OUT = config.ROOT / "web" / "src" / "lib" / "skeletons.json"


def main():
    out = {}
    for letter in config.LETTERS:
        with np.load(config.PROCESSED_DIR / f"{letter}.npz") as d:
            hands = normalize(d["landmarks"][~d["enhanced"]]).reshape(-1, config.NUM_LANDMARKS, 3)[:, :, :2]
        typical = hands[np.argmin(np.linalg.norm((hands - np.median(hands, axis=0)).reshape(len(hands), -1), axis=1))]
        typical = typical * [-1, 1]  # mirror, like the camera preview
        out[letter] = [[round(float(x), 3), round(float(y), 3)] for x, y in typical]
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"Saved {OUT.relative_to(config.ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
