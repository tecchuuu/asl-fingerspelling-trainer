"""Check the confidence gate on fake frame sequences (no camera needed).

Run from the project root:  python -m scripts.check_gate
"""

import numpy as np

from src import config
from src.gate import Gate

PALM = 100  # px


def frame(letter, conf=0.95, wrist_x=300):
    """25 classes: the 24 letters + "other"."""
    n = config.NUM_CLASSES + 1
    probs = np.full(n, (1 - conf) / (n - 1))
    probs[config.OTHER_INDEX if letter == config.OTHER else config.LETTERS.index(letter)] = conf
    return probs, wrist_x


NO_HAND = (None, None)

CASES = {
    "hold A for 2 s types one A": ([frame("A")] * 60, "A"),
    "A then B": ([frame("A")] * 15 + [frame("B")] * 15, "AB"),
    "flicker between letters is ignored": ([frame("A")] * 15 + [frame("S"), frame("E"), frame("S")] + [frame("B")] * 15, "AB"),
    "9 stable frames is not enough": ([frame("A")] * 9, ""),
    "low confidence never types": ([frame("A", conf=0.6)] * 30, ""),
    "hand leaves, same letter again": ([frame("A")] * 15 + [NO_HAND] * 6 + [frame("A")] * 15, "AA"),
    "brief hand loss doesn't re-type": ([frame("A")] * 15 + [NO_HAND] * 3 + [frame("A")] * 15, "A"),
    "low-confidence blip doesn't re-type": ([frame("A")] * 15 + [frame("A", conf=0.5)] * 2 + [frame("A")] * 15, "A"),
    "slide 0.8 palm sideways = double": ([frame("L")] * 15 + [frame("L", wrist_x=380)] * 15, "LL"),
    "small drift 0.3 palm is not a double": ([frame("L")] * 15 + [frame("L", wrist_x=330)] * 15, "L"),
    "an open hand ('other') never types": ([frame("other")] * 60, ""),
    "other between two holds doesn't re-type": ([frame("A")] * 15 + [frame("other")] * 20 + [frame("A")] * 15, "A"),
    "other then a new letter": ([frame("other")] * 20 + [frame("B")] * 15, "B"),
    "HELLO": ([frame("H")] * 12 + [frame("E")] * 12 + [frame("L")] * 12 + [frame("L", wrist_x=370)] * 12
              + [frame("O")] * 12, "HELLO"),
}


def run(frames):
    gate = Gate()
    return "".join(gate.update(probs, wrist_x=x, palm=PALM) or "" for probs, x in frames)


def main():
    failed = 0
    for name, (frames, want) in CASES.items():
        got = run(frames)
        failed += got != want
        print(f"{'PASS' if got == want else 'FAIL'}  {name:<40} expected {want!r}, got {got!r}")
    print("All passed." if not failed else f"{failed} failed.")


if __name__ == "__main__":
    main()
