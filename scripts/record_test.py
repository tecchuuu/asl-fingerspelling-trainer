"""Record the webcam test set: for each letter, sign it and press SPACE to capture.

Saves landmarks (for the MLP) and a 64x64 hand crop (for the CNN) per capture.
Never used for training. Record before looking at webcam results, and don't re-record
a letter because a model got it wrong.

Run from the project root, one session per recording condition:
    python -m scripts.record_test --session day1-window-right --hand right --notes "daylight, plain wall, ~60 cm"

Keys:  SPACE capture   U undo last   Q / Esc save and quit (rerun the same session to continue)
"""

import argparse
import json
import sys
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)

from src import config
from src.extract import crop_hand

REPEATS = 3
CAMERA_INDEX = 0
COOLDOWN_S = 0.8

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)


def session_path(name):
    return config.WEBCAM_TEST_DIR / f"{name}.npz"


def load_session(name):
    path = session_path(name)
    if not path.exists():
        return {"landmarks": [], "crops": [], "labels": [], "handedness": [], "frame_size": []}
    with np.load(path) as data:
        return {
            "landmarks": list(data["landmarks"]),
            "crops": list(data["crops"]),
            "labels": list(data["labels"]),
            "handedness": list(data["handedness"]),
            "frame_size": list(data["frame_size"]),
        }


def save_session(name, samples, meta):
    config.WEBCAM_TEST_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        session_path(name),
        landmarks=np.array(samples["landmarks"], dtype=np.float32).reshape(-1, config.NUM_LANDMARKS, 3),
        crops=np.array(samples["crops"], dtype=np.uint8).reshape(-1, config.CROP_SIZE, config.CROP_SIZE, 3),
        labels=np.array(samples["labels"], dtype=str),
        handedness=np.array(samples["handedness"], dtype=str),
        frame_size=np.array(samples["frame_size"], dtype=np.int32).reshape(-1, 2),
    )
    meta_path = config.WEBCAM_TEST_DIR / f"{name}.json"
    meta_path.write_text(json.dumps({**meta, "samples": len(samples["labels"])}, indent=2))


def queue_from(samples, letters, repeats):
    """Letters still to record, in a fixed shuffled order (same order when a session is resumed)."""
    order = list(np.random.default_rng(config.SEED).permutation(letters))
    done = {l: samples["labels"].count(l) for l in letters}
    return [l for l in order for _ in range(repeats - done[l])]


def open_camera():
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    # MJPG uses less USB bandwidth than raw frames, which makes some webcams drop out less
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def reconnect(cap):
    cap.release()
    for _ in range(3):
        cap = open_camera()
        ok, frame = cap.read()
        if ok:
            print("Camera back.")
            return cap, frame
        cap.release()
        time.sleep(1)
    return cap, None


class CameraReader:
    """Reads frames and recovers from the two ways this webcam fails:
    returning nothing, and silently repeating the same frozen frame."""

    STALE_FRAMES = 45  # ~1.5 s of identical frames; a live sensor always adds a little noise

    def __init__(self, cap):
        self.cap = cap
        self.prev = None
        self.stale = 0

    def read(self):
        for _ in range(30):
            ok, frame = self.cap.read()
            if ok:
                break
            time.sleep(0.05)
        else:
            print("Camera dropped out, reconnecting...")
            return self._reconnect()

        small = frame[::8, ::8]
        self.stale = self.stale + 1 if self.prev is not None and np.array_equal(small, self.prev) else 0
        self.prev = small
        if self.stale >= self.STALE_FRAMES:
            print("Camera image frozen, reconnecting...")
            return self._reconnect()
        return frame

    def _reconnect(self):
        self.cap, frame = reconnect(self.cap)
        self.prev, self.stale = None, 0
        return frame


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, points[conn.start], points[conn.end], GREEN, 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, RED, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="short name, e.g. day1-window-right")
    parser.add_argument("--hand", choices=["right", "left"], required=True)
    parser.add_argument("--notes", default="", help="lighting, background, distance")
    parser.add_argument("--letters", nargs="+", default=config.LETTERS, help="only these letters, e.g. T K P")
    parser.add_argument("--repeats", type=int, default=REPEATS, help="captures per letter")
    args = parser.parse_args()

    samples = load_session(args.session)
    unknown = set(args.letters) - set(config.LETTERS)
    if unknown:
        parser.error(f"unknown letters: {sorted(unknown)}")
    queue = queue_from(samples, args.letters, args.repeats)
    if not queue:
        print(f"Session '{args.session}' is already complete ({len(samples['labels'])} samples).")
        return
    meta = {"session": args.session, "hand": args.hand, "notes": args.notes,
            "started": datetime.now().isoformat(timespec="seconds")}

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.HAND_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
    )
    cap = open_camera()
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return
    camera = CameraReader(cap)

    total = len(args.letters) * args.repeats
    message, message_color = "", WHITE
    last_capture = 0.0
    last_ts = -1

    with HandLandmarker.create_from_options(options) as landmarker:
        while queue:
            frame = camera.read()
            if frame is None:
                print("Camera stopped sending frames and didn't come back.")
                break

            # Detect on the raw camera frame (what gets saved); only the preview is mirrored
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(int(time.monotonic() * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            preview = frame.copy()
            if result.hand_landmarks:
                draw_hand(preview, result.hand_landmarks[0])
            preview = cv2.flip(preview, 1)

            letter = queue[0]
            rep = samples["labels"].count(letter) + 1
            cooling = time.monotonic() - last_capture < COOLDOWN_S
            cv2.putText(preview, f"Sign: {letter}  ({rep}/{args.repeats})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, YELLOW, 3)
            cv2.putText(preview, f"{len(samples['labels'])}/{total} captured   SPACE capture  U undo  Q quit",
                        (10, preview.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)
            if cooling:
                cv2.putText(preview, "relax your hand...", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            elif message:
                cv2.putText(preview, message, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, message_color, 2)
            cv2.imshow("Record webcam test set", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u") and samples["labels"]:
                undone = samples["labels"][-1]
                for k in samples:
                    samples[k].pop()
                queue = queue_from(samples, args.letters, args.repeats)
                message, message_color = f"Undid last {undone}", WHITE
            if key == ord(" ") and not cooling:
                if not result.hand_landmarks:
                    message, message_color = "No hand found - try again", RED
                    continue
                hand = result.hand_landmarks[0]
                samples["landmarks"].append([(lm.x, lm.y, lm.z) for lm in hand])
                samples["crops"].append(crop_hand(rgb, hand))
                samples["labels"].append(letter)
                samples["handedness"].append(result.handedness[0][0].category_name)
                samples["frame_size"].append(rgb.shape[1::-1])  # (width, height)
                queue.pop(0)
                last_capture = time.monotonic()
                message, message_color = f"Saved {letter}", GREEN
                save_session(args.session, samples, meta)

    camera.cap.release()
    cv2.destroyAllWindows()
    if not samples["labels"]:
        print(f"Nothing captured, no session file written for '{args.session}'.")
        return
    save_session(args.session, samples, meta)
    status = "complete" if not queue else f"{len(queue)} left, rerun the same command to continue"
    print(f"Session '{args.session}': {len(samples['labels'])}/{total} samples saved ({status}).")
    print(f"  {session_path(args.session).relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
