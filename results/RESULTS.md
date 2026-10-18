# Results

Raw numbers for every training run are in `runs.csv`. This file records findings and decisions.

## 1. Data

- Dataset: ASL Alphabet (Kaggle, grassknoted), 25 classes kept (24 static letters + `nothing`)
- Images per class: 3,000 (75,000 for the 25 kept classes). The provided test folder has only 1 image per class, so the test split comes from the train folder.
- MediaPipe failures (no hand found):
  - Letter A check (`scripts/check_letter.py`, all 3,000 images): 818 failed (27.3%). Failures come in runs of consecutive frames and are mostly backlit shots (dark hand against a bright window). See `figures/check_A.png`.
  - Detected skeletons line up with the hand. Coordinates are in range (x, y ≈ 0..1, z relative to the wrist).
  - Speed: 20 ms/image, about 25 min for all 75,000 images.
- Class balance: perfectly balanced before MediaPipe failures are dropped

## 2. Setup decisions

- Mirroring: <!-- result of the webcam test: mirror ON/OFF, does Left/Right match the real hand? -->
- Normalization:
- Train / val / test split:

## 3. Tuning (MLP)

| Optimizer | Batch size | Val F1 | Epochs | Notes |
|---|---|---|---|---|
| Adam (1e-3) | 32 | | | |
| Adam (1e-3) | 64 | | | |
| SGD + momentum (1e-2, 0.9) | 32 | | | |
| SGD + momentum (1e-2, 0.9) | 64 | | | |

Winner:

Follow-ups on the winner (LR schedule, dropout 0.3 vs 0.5):

## 4. MLP vs CNN

| Model | Dataset test F1 | Webcam test F1 | Size | ms / frame |
|---|---|---|---|---|
| A: CNN (hand crops) | | | | |
| B: MLP (landmarks) | | | | |

## 5. Live app

- Gate settings:
- Double-letter rule:
- End-to-end ms per frame:

## 6. Problems and fixes

- **Webcam stopped sending frames on Windows.** OpenCV's default backend (MSMF) dropped the stream after a few seconds. Fixed by opening the camera with DirectShow (`cv2.CAP_DSHOW`).

## 7. Report lines
