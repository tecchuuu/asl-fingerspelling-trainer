"""Live MediaPipe hand skeleton for checking tracking, lighting and mirroring."""

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
CAMERA_INDEX = 0
NUM_HANDS = 2

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, points[conn.start], points[conn.end], GREEN, 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, RED, -1)
    return points


def print_landmarks(result):
    if not result.hand_landmarks:
        print("No hand in this frame.")
        return
    label = result.handedness[0][0].category_name
    print(f"\n{label} hand:")
    for i, lm in enumerate(result.hand_landmarks[0]):
        print(f"  {i:2d}: x={lm.x:.3f}  y={lm.y:.3f}  z={lm.z:+.3f}")


def main():
    if not MODEL_PATH.exists():
        print(f"Model file not found: {MODEL_PATH}\nDownload it from:\n  {MODEL_URL}")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=NUM_HANDS,
    )

    # The default Windows backend (MSMF) drops the stream on some webcams
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return

    mirror = True
    last_ts = -1
    fps = 0.0
    prev_time = time.perf_counter()

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped sending frames.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO mode rejects timestamps that don't strictly increase
            ts = max(int(time.monotonic() * 1000), last_ts + 1)
            last_ts = ts

            t0 = time.perf_counter()
            result = landmarker.detect_for_video(mp_image, ts)
            detect_ms = (time.perf_counter() - t0) * 1000

            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                points = draw_hand(frame, landmarks)
                category = handedness[0]
                wrist_x, wrist_y = points[0]
                cv2.putText(frame, f"{category.category_name} {category.score:.2f}",
                            (wrist_x - 40, wrist_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1 / (now - prev_time))
            prev_time = now

            lines = [
                f"FPS: {fps:.0f}   detect: {detect_ms:.1f} ms",
                f"Hands: {len(result.hand_landmarks)}   mirror: {'ON' if mirror else 'OFF'}",
                "m: mirror   p: print landmarks   q: quit",
            ]
            for i, text in enumerate(lines):
                cv2.putText(frame, text, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

            cv2.imshow("Webcam test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mirror = not mirror
            if key == ord("p"):
                print_landmarks(result)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
