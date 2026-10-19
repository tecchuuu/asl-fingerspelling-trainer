"""Record non-letter hand poses for the MLP's "other" class (TRAINING data, not test).

Press SPACE to start capturing, then keep moving your hand through poses that are NOT
letters: open spread hand, relaxed or half-closed hand, hand turning, waving, reaching,
hand partly out of frame. Avoid forming real letters (flat B, fist A/S, V, L, ...).
A capture is taken every few frames while a hand is visible.

Run from the project root:
    python -m scripts.record_other --session other-1 --notes "room light, right hand"
    python -m scripts.record_other --session other-test --target 200 --notes "..."

Keys:  SPACE start / pause   Q / Esc save and quit (rerun the same session to add more)
"""

import argparse
import json
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from src import config
from src.camera import CAMERA_INDEX, GREEN, RED, WHITE, YELLOW, CameraReader, draw_hand, open_camera, window_closed

WINDOW = "Record non-letter poses"
CAPTURE_EVERY = 6  # frames, ~5 captures per second


def path_of(session):
    return config.OTHER_DIR / f"{session}.npz"


def load(session):
    if not path_of(session).exists():
        return [], []
    with np.load(path_of(session)) as d:
        return list(d["landmarks"]), list(d["frame_size"])


def save(session, landmarks, frame_sizes, notes):
    config.OTHER_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path_of(session),
             landmarks=np.array(landmarks, dtype=np.float32).reshape(-1, config.NUM_LANDMARKS, 3),
             frame_size=np.array(frame_sizes, dtype=np.int32).reshape(-1, 2))
    (config.OTHER_DIR / f"{session}.json").write_text(json.dumps({"session": session, "notes": notes,
                                                                  "samples": len(landmarks)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--target", type=int, default=600, help="captures to aim for (counter turns green)")
    args = parser.parse_args()

    landmarks, frame_sizes = load(args.session)
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

    capturing = False
    frame_no = 0
    last_ts = -1
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            frame = camera.read()
            if frame is None:
                print("Camera stopped sending frames and didn't come back.")
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(int(time.monotonic() * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            frame_no += 1
            if capturing and result.hand_landmarks and frame_no % CAPTURE_EVERY == 0:
                landmarks.append([(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]])
                frame_sizes.append(rgb.shape[1::-1])

            preview = frame.copy()
            if result.hand_landmarks:
                draw_hand(preview, result.hand_landmarks[0])
            preview = cv2.flip(preview, 1)
            status, color = ("CAPTURING - keep moving, no letters", RED) if capturing else ("PAUSED - SPACE to start", YELLOW)
            cv2.putText(preview, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(preview, f"{len(landmarks)} / {args.target} captured", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN if len(landmarks) >= args.target else WHITE, 2)
            cv2.putText(preview, "SPACE start/pause   Q quit", (10, preview.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)
            cv2.imshow(WINDOW, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or window_closed(WINDOW):
                break
            if key == ord(" "):
                capturing = not capturing
                save(args.session, landmarks, frame_sizes, args.notes)

    camera.release()
    cv2.destroyAllWindows()
    if landmarks:
        save(args.session, landmarks, frame_sizes, args.notes)
    print(f"Session '{args.session}': {len(landmarks)} samples in {path_of(args.session).relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
