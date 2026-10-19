"""Send recorded webcam samples through the API and check it agrees with calling the model directly.

Run from the project root (no server needed, uses FastAPI's test client):
    python -m scripts.check_api
"""

import time

import numpy as np
from fastapi.testclient import TestClient

from app.main import CLASSES, app, model
from src import config
from src.predict import letter_probs

SESSION = "room-right"


def main():
    client = TestClient(app)
    print("GET /health ->", client.get("/api/health").json())

    with np.load(config.WEBCAM_TEST_DIR / f"{SESSION}.npz") as d:
        landmarks, labels, sizes = d["landmarks"], d["labels"], d["frame_size"]

    agree = correct = 0
    times = []
    for lm, label, (w, h) in zip(landmarks, labels, sizes):
        t0 = time.perf_counter()
        r = client.post("/api/predict", json={"landmarks": lm.tolist(), "width": int(w), "height": int(h)})
        times.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        api = r.json()
        direct = CLASSES[int(letter_probs(model, lm, aspect=w / h).argmax())]
        agree += api["letter"] == direct
        correct += api["letter"] == str(label)

    n = len(labels)
    print(f"{SESSION}: {n} samples | API == direct model: {agree}/{n} | API correct: {correct}/{n}")
    print(f"Request time (in-process, no network): median {np.median(times):.2f} ms")
    print("Example response:", client.post("/api/predict", json={"landmarks": landmarks[0].tolist(),
                                                              "width": int(sizes[0][0]), "height": int(sizes[0][1])}).json())
    bad = client.post("/api/predict", json={"landmarks": [[0, 0, 0]] * 5, "width": 640, "height": 480})
    print(f"Malformed request (5 points) -> HTTP {bad.status_code} (expected 422)")


if __name__ == "__main__":
    main()
