"""Webcam helpers shared by the recorder and the live demo."""

import sys
import time

import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections

CAMERA_INDEX = 0

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)


def open_camera():
    # The default Windows backend (MSMF) drops the stream on this webcam
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

    def release(self):
        self.cap.release()


def window_closed(name):
    """True once the user clicks the window's X (OpenCV doesn't report that as a key press)."""
    return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, points[conn.start], points[conn.end], GREEN, 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, RED, -1)
