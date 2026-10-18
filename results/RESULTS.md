# Results

Raw numbers for every training run are in `runs.csv`. This file records findings and decisions.

## 1. Data

- Dataset: ASL Alphabet (Kaggle, grassknoted). Models classify the 24 static letters.
  - `nothing` dropped as a model class: MediaPipe found no hand in 300/300 `nothing` images, so it has no landmarks or crops. In the app, "no hand detected" plays the role of `nothing`.
  - Dataset photos have a ~2 px blue frame. Trimmed before detection and crops are black-padded, so the CNN can't use the frame as a cue that won't exist on the webcam.
- Images per class: 3,000 (72,000 for the 24 letters). The provided test folder has only 1 image per class, so the test split comes from the train folder.
- MediaPipe failures (no hand found):
  - Letter A check (`scripts/check_letter.py`, all 3,000 images): 818 failed (27.3%). Failures come in runs of consecutive frames and are mostly backlit shots (dark hand against a bright window). See `figures/check_A.png`.
  - Detected skeletons line up with the hand. Coordinates are in range (x, y ≈ 0..1, z relative to the wrist).
  - Speed: 20 ms/image, about 25 min for all 75,000 images.
  - First full run, no brightening (`src/extract.py`, 10 workers, 5.3 min): **55,689 / 72,000 found (22.7% failed)**. Per letter in `extraction_summary.csv`.
  - Worst: N 55.0%, M 45.4%, C 34.7%, P 31.0%, Q 30.4%. Best: F 4.4%, K 10.1%.
  - M and N are closed-fist letters with the thumb tucked under fingers, the hardest shapes for MediaPipe. Fewest samples left: N 1,349 vs F 2,867 (2.1× imbalance).
- Brightening retry on the 16,311 failed images (`scripts/try_brightening.py`, per letter in `brightening_experiment.csv`):

  | Method | Recovered |
  |---|---|
  | CLAHE (local contrast) | 2.1% |
  | Gamma 0.5 (brighten shadows) | **43.0%** |
  | CLAHE + gamma | 2.8% |

  - CLAHE barely helps: the backlit hands are uniformly dark, so there's little local contrast to stretch. Gamma lifts the dark range directly.
  - M +369 and N +439 recovered (27% of their failures), the least of any letter: their failure is mostly the closed-fist shape, not light.
  - Sanity check: upside-down skeletons (wrist above the middle knuckle) are ~1% in recovered M/N and 0% in A/B/E/X, about the same as the originally detected samples (0.4% overall).
  - Examples: `figures/gamma_recovered.png`.
- **Final extraction (gamma retry built into `src/extract.py`): 62,704 / 72,000 found (12.9% failed)**, of which 7,015 via brightening. Each sample has an `enhanced` flag, so the models can be trained with and without them.
  - Still worst: N 40.4%, M 33.1%, X 22.3% failed. Best: F 0.8%, D 4.6%.
- Class balance: perfectly balanced before MediaPipe failures are dropped. After: 1,788 (N) to 2,977 (F) per letter, a 1.7× imbalance (2.1× without brightening).

## 2. Setup decisions

- Mirroring (webcam): <!-- result of the webcam test: mirror ON/OFF, does Left/Right match the real hand? -->
- Normalization (`src/normalize.py`, used by training and the app): 21×3 raw landmarks → 63 features.
  1. **Aspect fix:** x and z × (width / height). MediaPipe divides x by width and y by height, so a 640×480 webcam frame would stretch the hand horizontally vs the square dataset photos.
  2. **Translation:** subtract the wrist → wrist at (0, 0, 0), position in the frame no longer matters.
  3. **Scale:** divide by palm length (wrist → middle-finger knuckle). Chosen over the bounding box because palm length doesn't change with finger pose (a fist and an open hand have the same palm, different boxes).
  4. **Rotation is NOT normalized on purpose:** orientation is part of some letters (P/Q point down, G/H point sideways, vs K/U pointing up).
  - Checks (`scripts/check_normalize.py`): moving, resizing or aspect-squashing a hand gives identical features; mirroring twice is a no-op. All 62,704 samples: wrist = 0, palm length = 1.
  - Outliers: only 35 samples (0.06%) have a point more than 3 palm lengths from the wrist (mostly C), so no filtering.
  - The 3 wrist features are always 0 (63 inputs, 60 informative). Kept for simplicity.
  - **Effect:** nearest-centroid classifier (one average per letter, no training) goes from **50.4% on raw landmarks to 85.4% on normalized**. Figure: `figures/normalize_before_after.png` (40 'A' hands before/after).
- Handedness: **mirror augmentation, not label-based flipping.** The dataset is one signer, yet MediaPipe labels ~5% of hands "Left" (e.g. Y: 2,560 Right / 20 Left), so the label is too unreliable to decide which hands to flip. Instead, training adds a mirrored copy (`mirror()`: x → −x) so the model learns both hands.
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
