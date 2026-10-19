"""Web API: the browser sends MediaPipe hand landmarks, the server answers with the letter.

Run from the project root:
    python -m uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs to try the endpoints in the browser.
Use 127.0.0.1, not "localhost": on this Windows machine "localhost" tries IPv6 first and
waits ~2 s before falling back, which made every request take 2 s instead of 2 ms.
"""

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src import config
from src.evaluate_webcam import load_final
from src.predict import letter_probs

MODEL_NAME = "mlp_other"

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


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "classes": len(CLASSES)}


@app.post("/predict", response_model=Prediction)
def predict(hand: Hand):
    raw = np.asarray(hand.landmarks, dtype=np.float32)
    probs = letter_probs(model, raw, aspect=hand.width / hand.height)
    order = np.argsort(probs)[::-1]
    return Prediction(
        letter=CLASSES[order[0]],
        confidence=float(probs[order[0]]),
        top3=[(CLASSES[i], float(probs[i])) for i in order[:3]],
    )
