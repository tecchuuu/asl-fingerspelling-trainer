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
- Train / val / test split (`src/dataset.py`, checks in `scripts/check_split.py`): **block split, ~70 / 15 / 15 per letter.**
  - Frame numbers follow recording order: consecutive frames are ~4× closer in landmark space than random pairs (A: 0.24 vs 1.01), and frames ≤100 apart are closer than frames >1000 apart (0.71 vs 1.13).
  - So frames are grouped in blocks of 100 (≈30 blocks per letter) and whole blocks go to one split. Blocks are assigned until each split reaches its share of the letter's **samples** (not blocks), because failed detections cluster in certain blocks. First version split by block count and one letter got only a quarter of its fair test share.
  - Result: train 44,989 (71.7%), val 9,132 (14.6%), test 8,583 (13.7%). Each letter has 11.3–15.0% of its samples in test. 12.7% of test samples are brightening-recovered.
  - **Leakage evidence:** 1-nearest-neighbour (copies the label of the most similar training sample) scores 98.8% on a random split vs 96.4% on the block split. Blocks reduce leakage but don't remove it: it's one signer in a few sessions, so any split of this dataset stays optimistic. That's why the webcam test exists.
  - Mirror augmentation applies to train only (44,989 → 89,978). Crops use the same split and are flipped horizontally.
  - `include_enhanced=False` removes brightened samples from **train only**; val/test stay identical so the with/without comparison is fair.

## 3. First MLP (Phase 1)

- Model (`src/models.py`): 63 → 256 → 128 → 24, ReLU, dropout 0.3 after each hidden layer. **52,376 parameters.**
  - ReLU: standard default, no vanishing gradients at this depth. Dropout: 60k samples of one signer overfit easily.
  - Outputs raw scores (logits); `CrossEntropyLoss` applies softmax itself.
- Training (`src/train.py`): Adam lr 1e-3, batch 32, max 50 epochs, early stopping when val loss doesn't improve for 5 epochs, best-val-loss checkpoint kept.
  - Every run appends a row to `runs.csv` and writes per-epoch curves to `history/<run_id>.csv`.
  - `train.py` only reports **validation** scores. The test set is touched once, for final models, so tuning can't overfit it.
- **Result (run `20261018-203217_mlp_adam_bs32`): val macro F1 96.5%, val accuracy 96.8%** at epoch 9 (stopped at 14), 108 s on CPU.
  - Already above the 90% dataset target. Expected: this split is optimistic (one signer, see the 1-NN leakage check).
  - Train accuracy 97.1% vs val 96.8% at the best epoch: no real overfitting.
  - Val loss jumps between epochs (0.12 → 0.18 → 0.12): batch 32 gives noisy updates. Worth comparing with batch 64 in Phase 2.

## 4. Tuning (MLP)

Same model (dropout 0.3), same split, seed 42, early stopping patience 5. Figure: `figures/optimizer_comparison.png` (made by `scripts/plot_curves.py`).

| Optimizer | Batch size | Val F1 | Val loss | Best epoch / run | Time |
|---|---|---|---|---|---|
| Adam (1e-3) | 32 | 96.45% | 0.1175 | 9 / 14 | 108 s |
| **Adam (1e-3)** | **64** | **96.60%** | **0.1133** | 20 / 25 | 87 s |
| SGD + momentum (1e-2, 0.9) | 32 | 96.37% | 0.1204 | 15 / 20 | 88 s |
| SGD + momentum (1e-2, 0.9) | 64 | 96.44% | 0.1161 | 30 / 35 | 101 s |

- **All four end within 0.25 points of F1.** With one seed each, that gap is about the size of run-to-run noise, so the optimizers reach the same quality on this problem.
- **The real difference is how fast they get there.** Adam is at ~94% F1 by epoch 2–3; SGD bs64 needs ~8 epochs for the same and ~30 to reach its best. Adam adapts the step size per weight; plain SGD with one global lr has to crawl.
- **Batch 64 is smoother than 32** for both optimizers (fewer val-loss spikes, e.g. Adam bs32's jump at epoch 8). Larger batches average out more noise per step. It also trains faster per epoch on CPU (fewer, bigger steps).
- No overfitting in any run: best-epoch train loss (0.07–0.10) stays close to val loss (0.11–0.12).

Winner: **Adam, lr 1e-3, batch 64**: best val F1 and val loss, smooth curve, fastest wall-clock. Margin is small, so the honest claim is "Adam converges much faster for the same final quality", not "Adam is more accurate".

### Follow-ups on the winner (Adam, lr 1e-3, batch 64)

Figure: `figures/schedule_dropout.png`.

| Run | Val F1 | Val loss | Train loss | Best epoch / run | Time |
|---|---|---|---|---|---|
| Fixed lr, dropout 0.3 | 96.60% | 0.1133 | 0.073 | 20 / 28 | 87 s |
| **ReduceLROnPlateau, dropout 0.3** | **96.81%** | **0.1013** | 0.034 | 44 / 50 | 157 s |
| ReduceLROnPlateau, dropout 0.5 | 96.51% | 0.1165 | 0.078 | 45 / 50 | 158 s |

All three use early-stopping patience 8 (see below).

- **LR schedule** (ReduceLROnPlateau: halve the lr when val loss hasn't improved for 2 epochs). Why try it: the fixed-lr curves keep bouncing after epoch ~10 (val loss 0.11–0.16), the sign of steps too large to settle.
  - LR halved at epochs 19, 26, 30, 40, 43, 48 (1e-3 → 1.6e-5).
  - **Val loss −11% (0.1133 → 0.1013)** and the late spikes disappear. F1 +0.2 points: small, the model was already near its ceiling.
  - Runs are deterministic: with the same seed, both runs are identical up to epoch 18, so the difference comes only from the schedule.
- **Scheduler vs early stopping interaction.** First attempt used patience 5: the lr dropped at epoch 19 and early stopping ended the run at epoch 20, before the smaller lr could help (best val loss missed by 0.0001). Lesson: early-stopping patience must be longer than the scheduler's patience. Reran fixed and schedule with patience 8 for a fair comparison.
- **Dropout 0.5 vs 0.3** (on top of the schedule). Why try it: with the schedule, train loss falls to 0.034 vs val 0.101, a widening gap, so more regularization might help.
  - Result: **worse** (F1 96.51%, val loss 0.1165) and train loss back up to 0.078. 0.5 drops half the neurons in a small 256/128 network: it underfits instead of generalizing better. Keep 0.3.

**Final MLP settings: Adam, lr 1e-3, batch 64, ReduceLROnPlateau (factor 0.5, patience 2), dropout 0.3, early-stopping patience 8, max 50 epochs.** Best run: `20261018-222208_mlp_adam_bs64`. These are the settings the CNN will use.

## 5. MLP vs CNN

### CNN design (`src/models.py`)

- Input: 64×64 RGB hand crop (same split as the MLP; mirrored crops added to train).
- 3 conv blocks: Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2. Each block halves the image and doubles the channels: 3×64×64 → 32×32×32 → 64×16×16 → 128×8×8.
  - 3×3 kernels: small local patterns (edges → finger shapes → hand shape as blocks stack).
  - BatchNorm: rescales each layer's outputs during training, so conv nets train stably with a normal learning rate.
  - MaxPool: shrinks the image and keeps the strongest response, so later layers see larger areas.
- Head: global average pool (each of 128 channels → 1 number) → dropout 0.3 → Linear → 24. Averaging instead of flattening keeps the parameter count small.
- **96,792 parameters** vs the MLP's 52,376: same order of size, so the comparison is about the input (pixels vs landmarks), not model size.
- Trained with the MLP's winning settings (Adam, lr 1e-3, batch 64, ReduceLROnPlateau, patience 8, max 50 epochs). No separate tuning grid for the baseline.
- CPU cost: **~236 s per epoch** (vs ~3 s for the MLP), so up to ~3.3 h per run. First epoch: val F1 67.1% (MLP's first epoch: 88.9%).

### First CNN run (validation)

Run `20261018-232701_cnn_adam_bs64`. Figure: `figures/mlp_vs_cnn.png`.

| | MLP (landmarks) | CNN (crops) |
|---|---|---|
| Val macro F1 | **96.81%** | 95.16% |
| Val loss | **0.101** | 0.134 |
| Train acc at best epoch | 99.0% | 99.7% |
| Val acc at best epoch | 97.1% | 95.5% |
| Train − val gap | **1.9 pts** | 4.2 pts |
| Parameters | 52,376 | 96,792 |
| Training time (CPU) | **157 s** | 9,154 s (2.5 h) |
| Best epoch / run | 44 / 50 | 32 / 40 |

- **The MLP wins on the dataset already**, by 1.7 F1 points, with half the parameters and ~58× less training time.
- **The CNN overfits more:** it reaches 99.7% on train but 95.5% on val (gap 4.2 pts vs the MLP's 1.9). It can memorize pixel details of this one signer and room; the MLP never sees them.
- **The CNN learns slower and noisier:** 67% F1 after epoch 1 (MLP 89%), val loss spikes until ~epoch 20 (e.g. epoch 19: 0.35). It must learn from pixels what MediaPipe gives the MLP directly.
- LR halvings at epochs 10, 21, 25, 30, 36, 39; early stopping at 40.
- **Expectation for the webcam test:** the gap should grow. The CNN's errors come from appearance (lighting, background, skin), which changes on a new camera; landmarks don't carry appearance at all.
- Both numbers are validation scores on the same block split. Test scores come later from `evaluate.py`, once, on the final models.

| Model | Dataset test F1 | Webcam test F1 | Size | ms / frame |
|---|---|---|---|---|
| A: CNN (hand crops) | | | | |
| B: MLP (landmarks) | | | | |

## 6. Live app

- Gate settings:
- Double-letter rule:
- End-to-end ms per frame:

## 7. Problems and fixes

- **Webcam stopped sending frames on Windows.** OpenCV's default backend (MSMF) dropped the stream after a few seconds. Fixed by opening the camera with DirectShow (`cv2.CAP_DSHOW`).

## 8. Report lines
