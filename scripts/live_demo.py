"""Live fingerspelling: webcam -> MediaPipe -> MLP -> confidence gate -> typed text.

Also measures the time per frame, so the "<= 33 ms per frame" target can be checked.

Run from the project root:  python -m scripts.live_demo
Keys:  BACKSPACE delete last letter   C clear   Q / Esc quit
"""

import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from src import config
from src.camera import CAMERA_INDEX, GREEN, WHITE, YELLOW, CameraReader, draw_hand, open_camera, window_closed
from src.evaluate_webcam import load_final
from src.gate import Gate
from src.predict import letter_probs

BACKSPACE = 8
WINDOW = "Live fingerspelling"


def palm_length_px(hand, w, h):
    wrist, mcp = hand[0], hand[9]
    return float(np.hypot((mcp.x - wrist.x) * w, (mcp.y - wrist.y) * h))


def draw_panel(frame, guess, conf, progress, text, timing):
    h, w = frame.shape[:2]
    if guess:
        color = GREEN if conf >= 0.8 else WHITE
        cv2.putText(frame, f"{guess}  {conf:.0%}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        cv2.rectangle(frame, (10, 62), (10 + int(200 * progress), 72), YELLOW, -1)
        cv2.rectangle(frame, (10, 62), (210, 72), WHITE, 1)
    else:
        cv2.putText(frame, "no hand", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)

    cv2.putText(frame, "BACKSPACE delete  C clear  Q quit", (w - 330, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, text[-22:] + "_", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2)
    cv2.putText(frame, timing, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def main():
    # The 25-class MLP knows "other" (a hand that isn't signing); fall back to the 24-letter one
    name = "mlp_other" if (config.ROOT / "models" / "mlp_other_final.pt").exists() else "mlp"
    model = load_final(name)
    print(f"Model: {name}")
    gate = Gate()
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

    text = ""
    last_ts = -1
    times = {"mediapipe": [], "model": [], "total": []}
    prev_frame = time.perf_counter()
    fps = 0.0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            frame = camera.read()
            if frame is None:
                print("Camera stopped sending frames and didn't come back.")
                break
            t0 = time.perf_counter()
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(int(time.monotonic() * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            t1 = time.perf_counter()

            guess, conf = None, 0.0
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                raw = np.array([(lm.x, lm.y, lm.z) for lm in hand], dtype=np.float32)
                probs = letter_probs(model, raw, aspect=w / h)
                t2 = time.perf_counter()
                top = int(probs.argmax())
                guess = config.OTHER if top == config.OTHER_INDEX else config.LETTERS[top]
                conf = float(probs.max())
                typed = gate.update(probs, wrist_x=hand[0].x * w, palm=palm_length_px(hand, w, h))
                times["model"].append((t2 - t1) * 1000)
            else:
                typed = gate.update(None)
            if typed:
                text += typed
            t_end = time.perf_counter()
            times["mediapipe"].append((t1 - t0) * 1000)
            times["total"].append((t_end - t0) * 1000)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / (now - prev_frame)
            prev_frame = now

            preview = frame.copy()
            if result.hand_landmarks:
                draw_hand(preview, result.hand_landmarks[0])
            preview = cv2.flip(preview, 1)
            timing = (f"FPS {fps:.0f}  |  MediaPipe {times['mediapipe'][-1]:.1f} ms  "
                      f"MLP {times['model'][-1] if result.hand_landmarks else 0:.2f} ms  "
                      f"total {times['total'][-1]:.1f} ms")
            draw_panel(preview, guess, conf, gate.progress, text, timing)
            cv2.imshow(WINDOW, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or window_closed(WINDOW):
                break
            if key == BACKSPACE:
                text = text[:-1]
            if key == ord("c"):
                text = ""

    camera.release()
    cv2.destroyAllWindows()
    if times["total"]:
        print(f"\nTyped: {text!r}")
        print(f"Frames: {len(times['total'])}  (median, 95th percentile)")
        for name, values in times.items():
            if values:
                print(f"  {name:<10} {np.median(values):6.2f} ms  {np.percentile(values, 95):6.2f} ms")


if __name__ == "__main__":
    main()
