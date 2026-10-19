"""Web API + app: the browser sends MediaPipe hand landmarks, the server answers with the letter.

Development (two terminals, from the project root):
    python -m uvicorn app.main:app --reload        # API on http://127.0.0.1:8000
    cd web && npm run dev                          # React app on http://127.0.0.1:5173 (proxies /api)
Demo (one terminal): cd web && npm run build, then only the uvicorn command; open http://127.0.0.1:8000

Use 127.0.0.1, not "localhost": on this Windows machine "localhost" tries IPv6 first and
waits ~2 s before falling back, which made every request take 2 s instead of 2 ms.
"""

import base64
import csv
from datetime import datetime
from typing import Literal

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from pydantic import BaseModel, Field

from app import db
from src import config
from src.evaluate_webcam import load_final
from src.predict import letter_probs

MODEL_NAME = "mlp_other"
WEB_DIST = config.ROOT / "web" / "dist"
COMPARE_CSV = config.RESULTS_DIR / "browser_vs_python.csv"

app = FastAPI(title="ASL fingerspelling trainer")
model = load_final(MODEL_NAME)
CLASSES = config.LETTERS + [config.OTHER]


class Hand(BaseModel):
    # The 21 points exactly as MediaPipe gives them: x, y in 0..1 of the image, z relative to the wrist
    landmarks: list[list[float]] = Field(min_length=21, max_length=21)
    width: int = Field(gt=0, description="video width in pixels")
    height: int = Field(gt=0, description="video height in pixels")


class Prediction(BaseModel):
    letter: str          # one of the 24 letters, or "other" for a hand that isn't signing
    confidence: float
    top3: list[tuple[str, float]]


def predict_raw(raw, width, height):
    probs = letter_probs(model, np.asarray(raw, dtype=np.float32), aspect=width / height)
    order = np.argsort(probs)[::-1]
    return Prediction(
        letter=CLASSES[order[0]],
        confidence=float(probs[order[0]]),
        top3=[(CLASSES[i], float(probs[i])) for i in order[:3]],
    )


db.init()


class SessionStart(BaseModel):
    mode: Literal["practice", "exam"]
    category: str = Field(min_length=1, max_length=40)
    items_total: int | None = None


class Attempt(BaseModel):
    session_id: int
    item: str = Field(min_length=1, max_length=40)
    position: int = Field(ge=0)
    target: str = Field(min_length=1, max_length=5)
    typed: str = Field(min_length=1, max_length=5)
    confidence: float | None = None
    ms: int | None = Field(default=None, ge=0)


@app.post("/api/sessions")
def start_session(req: SessionStart):
    return {"session_id": db.start_session(req.mode, req.category, req.items_total)}


@app.post("/api/sessions/{session_id}/finish")
def finish_session(session_id: int):
    score = db.finish_session(session_id)
    return {"session_id": session_id, "score": score}


@app.post("/api/attempts")
def add_attempt(a: Attempt):
    db.add_attempt(a.session_id, a.item, a.position, a.target, a.typed, a.confidence, a.ms)
    return {"ok": True}


@app.get("/api/progress")
def progress():
    return db.progress()


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "classes": len(CLASSES)}


@app.post("/api/predict", response_model=Prediction)
def predict(hand: Hand):
    return predict_raw(hand.landmarks, hand.width, hand.height)


# --- Phase 4 check: does MediaPipe in the browser give the same hand as MediaPipe in Python? ---

_python_landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(config.HAND_MODEL_PATH)),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
))


class CompareRequest(Hand):
    image: str = Field(description="the same video frame as a base64 JPEG data URL")


@app.post("/api/debug/compare")
def compare(req: CompareRequest):
    """Run Python MediaPipe on the frame the browser used, and compare landmarks and predictions."""
    jpeg = base64.b64decode(req.image.split(",", 1)[-1])
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "could not decode image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = _python_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.hand_landmarks:
        return {"python_found_hand": False, "browser": predict_raw(req.landmarks, req.width, req.height)}

    py = np.array([(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]], dtype=np.float32)
    js = np.asarray(req.landmarks, dtype=np.float32)
    h, w = rgb.shape[:2]
    pixel_diff = np.abs((py[:, :2] - js[:, :2]) * [w, h]).mean()
    out = {
        "python_found_hand": True,
        "mean_xy_diff_px": float(pixel_diff),
        "max_z_diff": float(np.abs(py[:, 2] - js[:, 2]).max()),
        "browser": predict_raw(js, req.width, req.height),
        "python": predict_raw(py, w, h),
    }
    _log_comparison(out)
    return out


def _log_comparison(r):
    new = not COMPARE_CSV.exists()
    with open(COMPARE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["time", "browser_letter", "browser_conf", "python_letter", "python_conf",
                             "same_letter", "mean_xy_diff_px", "max_z_diff"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"),
                         r["browser"].letter, f"{r['browser'].confidence:.3f}",
                         r["python"].letter, f"{r['python'].confidence:.3f}",
                         r["browser"].letter == r["python"].letter,
                         f"{r['mean_xy_diff_px']:.2f}", f"{r['max_z_diff']:.4f}"])


# The built React app (web/dist) is served last, so /api routes win
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
