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

### Before vs after tuning (what the rubric's "Baseline" means)

The course brief (`CS461_Final_Project_ปรับปรุง2569.pdf`) defines the Baseline as the first basic model that runs and gives initial results (Week 12 check-in), and the final report must show results before/after tuning. So the baseline is the **first, untuned MLP**, not a second architecture. The CNN comparison (section 5) comes from this project's own proposal, not the rubric.

| | Setup | Val F1 | Val loss |
|---|---|---|---|
| **Baseline** (Phase 1) | Adam, batch 32, fixed lr, dropout 0.3, patience 5 | 96.45% | 0.1175 |
| Optimizer × batch size | Adam, batch 64 | 96.60% | 0.1133 |
| **Final tuned** | + ReduceLROnPlateau, patience 8 | **96.81%** | **0.1013** (−14%) |

Tuning techniques covered, matching the brief's list: optimizer comparison (Adam vs SGD+momentum), batch size (32 vs 64), learning-rate scheduling (ReduceLROnPlateau vs fixed), early stopping, regularization (dropout 0.3 vs 0.5).

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

### Test set (`src/evaluate.py`, run once on the final models)

Final models chosen by **validation** F1, copied to `models/mlp_final.pt` and `models/cnn_final.pt`. Test set: 8,583 samples, same block split, same samples for both models (asserted). Numbers in `test_summary.csv` and `test_per_letter.csv`.

| Model | Dataset test F1 | Test acc | Webcam test F1 | Params | File | ms / sample (model only) |
|---|---|---|---|---|---|---|
| A: CNN (hand crops) | 96.02% | 96.17% | | 96,792 | 388 KB | 1.43 |
| B: MLP (landmarks) | **98.14%** | **98.33%** | | **52,376** | **208 KB** | **0.07** |

- **Both beat the 90% dataset target.** The MLP wins by 2.1 F1 points.
- **Test is higher than val for both** (MLP 98.1 vs 96.8, CNN 96.0 vs 95.2): the test blocks happen to be a bit easier than the val blocks. One block split = some luck in which recording blocks land where. It moves both models the same way, so the comparison holds; the absolute numbers carry roughly ±1–2 points of split luck.
- **Speed:** MLP 0.07 ms vs CNN 1.43 ms per sample (~20× faster), batch 1 on CPU, model only. Both are tiny next to MediaPipe's ~20 ms per frame, which dominates the live pipeline either way. The CNN also needs a crop step the MLP doesn't.

**Where each model fails** (figures: `figures/confusion_matrices.png`, `figures/per_letter_f1.png`):

| | MLP | CNN |
|---|---|---|
| Hardest letters (F1) | M 91.8%, N 93.8%, S 97.6% | H 87.0%, A 87.4%, G 89.9% |
| Top confusion | N → M (5.9% of N) | H → G (18.6% of H) |
| Others | Q → P 3.2%, S → I 2.5%, M → N 2.8% | A → L 9.2%, V → W 6.6%, I → R 5.9% |

- **The models fail on different letters.**
- **MLP: M and N**, the predicted closed-fist letters. M and N differ only in how many fingers fold over the tucked thumb (3 vs 2). In a fist, those fingertips overlap and MediaPipe has to guess hidden joints, so the landmarks of M and N come out similar. That's the same reason M/N had the most MediaPipe detection failures (33% / 40%).
- **CNN: H ↔ G.** Both point sideways; G extends the index finger, H the index + middle finger stacked on top of it. In a 64×64 crop the two stacked fingers look like one thick finger. Landmarks place each fingertip separately, so the MLP barely confuses them (H → G 1.7%).
- **The CNN is better on M and N** (94.0% / 94.8% vs the MLP's 91.8% / 93.8%): pixels still show the knuckle bumps that the landmarks lose. The two inputs carry complementary information. Future work: combine them (e.g. CNN only for M/N/S/T).

### Webcam test set: recording protocol (`scripts/record_test.py`)

- One signer (me). **Test only, never training.** Recorded before looking at any webcam results; letters are not re-recorded because a model got them wrong.
- Per session: 24 letters × 3 captures = 72. Letters come in a fixed shuffled order (not A→Y) so fatigue doesn't always hit the same letters. Relax the hand between captures (0.8 s cooldown), so the 3 captures aren't near-duplicates.
- Each capture saves the landmarks (MLP) and a 64×64 crop made by the same `crop_hand` as the dataset (CNN). Only the hand crop is stored, not the full frame.
- Detection runs on the **raw camera frame**, which is what gets saved; only the on-screen preview is mirrored, for comfort. Mirror augmentation in training covers either orientation.
- Frame size is saved per sample: webcam frames are 640×480, so the MLP gets `normalize(..., aspect=4/3)`.
- **Dataset handshape check** (spotted while recording; all 24 letters checked on 6 random training photos each, T/P/X also at full size; reference sheet `figures/letter_reference.png`). Judged by eye against standard ASL, not by an ASL expert — verify T against an ASL dictionary before citing it.
  - **T: non-standard.** Dataset = fingers curled, index half-raised, **thumb sticking out sideways**. Standard ASL T = fist with the thumb tucked between index and middle finger. Consistent across all photos checked.
  - **P: variant.** Index points out and thumb touches the middle finger (as in P), but the hand is horizontal instead of pointing down, and thumb + middle form a closed ring.
  - **K: variant** (spotted by me while recording). Dataset = V shape with the thumb lying across the front of the middle finger, low down. Standard ASL K = thumb tip pressed up between index and middle, against the middle finger's middle joint. Still distinct from V in the data (K test F1 ~99%).
  - **X: fine.** Hooked index finger in nearly all photos. (A first look at small, dark thumbnails wrongly suggested a straight index.)
  - Other 20 letters: consistent with standard ASL in the photos checked.
  - Decision: **in the main webcam sessions every letter is signed like the dataset reference sheet** (natural angle and position), so the test measures generalization to a new person/camera/lighting, not disagreement about handshapes. A separate `standard-asl` session will record the correct ASL version of the letters that differ (T, K, P, and any found later).
  - Consequence for the product: a trainer built on this data would accept the dataset's T and reject a correct ASL T, i.e. teach a wrong sign (confirmed by the standard-ASL check below; the app decision is logged there). Fix for a real product: re-record T, or use a signer-verified dataset. Possible extra test: a few standard-ASL T and P captures to measure how often the models reject them.
- Each session has a name and notes (lighting, background, distance, hand) in `data/webcam_test/<session>.json`. Planned conditions: daylight / night lamp, plain / busy background, near / far, right / left hand, different days.

### Webcam test: first 2 sessions (`src/evaluate_webcam.py`, overall scores only)

Scored after `room-right` and `screen-right`, before the left-hand sessions. Only overall F1 was looked at, no per-letter results, so the remaining sessions aren't biased toward "fixing" specific letters. Numbers in `webcam_summary.csv`.

| Session | n | MLP F1 | CNN F1 | MLP acc | CNN acc |
|---|---|---|---|---|---|
| room-right (room light) | 72 | **98.6%** | 17.4% | 98.6% | 25.0% |
| screen-right (dark, monitor light only) | 72 | **87.0%** | 0.8% | 88.9% | 4.2% |
| **Both** | 144 | **93.5%** | 12.5% | 93.8% | 14.6% |

- **MLP: above the 80% webcam target** (93.5%). In room light it matches its dataset test score (98.6% vs 98.1%); in the dark it drops ~11 points but stays usable.
- **CNN: collapses.** 25% accuracy in room light, 4.2% in the dark, which is chance level for 24 letters (1/24 = 4.2%). Its 96% dataset score doesn't carry over at all.
- **Not a bug:** the same scoring code gives the CNN 97.7% F1 on 1,000 dataset test crops, and the webcam crops are correctly framed hands (checked visually against dataset crops).
- **Why:** the webcam crops differ from the dataset in everything except hand shape: a different person's skin, a dark chair and curtains behind the hand instead of a plain bright wall, a higher camera angle, and in the screen session much darker images (mean pixel brightness: dataset 111, room-right 146, screen-right 60). The CNN learned what this one signer's hands in this one room look like; the MLP only ever saw finger geometry, which is the same on any camera.
- **This is the report's central result:** on the dataset the two models are 2 points apart (98.1 vs 96.0); on a real webcam they are ~80 points apart. Dataset accuracy alone would have picked either model; only the webcam test shows which one works.
- **Decision: no further CNN work.** The comparison has answered the proposal's question, and neither the app nor the rubric needs the CNN (the rubric's "Baseline" is the untuned MLP, see section 4). The CNN stays frozen as `models/cnn_final.pt` and is still scored automatically on every webcam session by `evaluate_webcam.py`, so the comparison stays complete.
  - Dropped, kept as future work: retraining the CNN with brightness/colour/background augmentation, a pretrained (ImageNet) backbone, extra CNN optimizer runs, a live side-by-side CNN demo.
  - In hindsight the CNN was optional scope: it tested whether the landmark approach was the right bet, which wasn't known when planning. It cost one script and one overnight run.
- Caveats: 72 samples per session, so ±a few points of noise; one signer. `screen-right` was partly captured rapid-fire (hand held, small adjustments) — to be checked when all sessions are in.

### Webcam test: final results (3 sessions + standard-ASL check)

All sessions recorded before any per-letter result was looked at. `screen-left` and the distance sessions were not recorded (dark lighting tested with the right hand only; distance not tested). Figure: `figures/webcam_vs_dataset.png`.

| Session | n | MLP F1 | CNN F1 | MLP acc | CNN acc |
|---|---|---|---|---|---|
| room-right (room light, right hand) | 72 | **98.6%** | 17.4% | 98.6% | 25.0% |
| room-left (room light, left hand) | 72 | **91.3%** | 18.1% | 93.1% | 27.8% |
| screen-right (dark, monitor light) | 72 | **87.0%** | 0.8% | 88.9% | 4.2% |
| **All webcam** | 216 | **93.1%** | 16.0% | 93.5% | 19.0% |
| *Dataset test (for comparison)* | 8,583 | *98.1%* | *96.0%* | | |

- **MLP meets the 80% webcam target in every session** (93.1% overall).
- **Hand effect:** left hand −7 points vs right (91.3 vs 98.6). Mirror augmentation mostly works; the remaining gap is likely the non-dominant hand signing less precisely, not mirroring itself (can't separate the two with one signer).
- **Lighting effect:** dark −12 points (87.0 vs 98.6). Usable, but the app should advise front lighting.
- **CNN:** 16% overall, chance level in the dark. Dataset test 96% → webcam 16%: the central finding (section 5).

**MLP webcam errors (all sessions, 9 samples per letter):**

| Letter | F1 | Main confusion |
|---|---|---|
| Q | 50% | Q → P, 6 of 9 |
| P | 75% | (receives the Q errors) |
| Y | 80% | Y → V, 3 of 9 |
| S | 80% | S → E, 3 of 9 |
| E, V | 86% | |

- **Q ↔ P is the main real-world weakness**, not M/N (the dataset's weakest pair). Both letters point down; with the camera mounted high and looking down, downward-pointing hands are seen from an angle the dataset (camera roughly level) doesn't have. On the dataset test Q → P was already the 2nd most common MLP confusion (3.2%); the webcam angle amplifies it.
- S → E: both are fists, differing in thumb position (across the fingers vs tucked under), which landmarks capture poorly in a fist.
- 9 samples per letter, so per-letter numbers are rough (one error = 11 points).

**Within-letter variation check** (median distance between the 3 captures of a letter, normalized features): room-right 0.54, room-left 0.63, screen-right 0.70. No letter in any session has near-duplicate captures (all > 0.15). So the partly rapid-fire `screen-right` session didn't produce near-duplicates; its spread is actually the largest (darker images → noisier landmarks).

**Standard-ASL check** (`standard-asl`: T, K, P signed per ASL dictionaries, 5 each, room light, right hand; not in the main score):

| Letter | MLP accepts | CNN accepts | MLP predicts instead |
|---|---|---|---|
| T | **0%** | 0% | A |
| K | 80% | 0% | W (1 of 5) |
| P | 100% | 0% | – |

- **T is a real problem:** the MLP never accepts the correct ASL T and reads it as A (both are fists with the thumb close to the index finger). A trainer built on this data would reject every correctly signed T.
  - Options considered: (a) exclude T from the app like J and Z; (b) keep T in the prototype, accepting the dataset's handshape, with a visible label.
  - **Decision: (b).** The app is a proof of concept, not deployed to learners. T stays, shown with a ⚠️ note: "Dataset version, not standard ASL. A real deployment needs T re-recorded with the correct handshape." This makes the data limitation visible instead of hiding it.
  - Before any real deployment: re-record T with the correct ASL handshape (ideally several signers) and retrain.
  - Fix options, ranked:
    1. **Stretch goal (only after the app works):** replace the T class with correct-T images from another Kaggle dataset (check its T by eye against an ASL dictionary first), run `extract.py` on them, retrain the MLP (~2.5 min), re-evaluate. The 5 standard-ASL T captures then become the before/after test (0/5 now). The CNN stays frozen.
    2. **Future work only:** add whole other datasets (all letters, more signers) for better real-world generalization.
    3. **Rejected:** replacing the entire dataset. It would redo extraction, tuning, the CNN and every evaluation, for a flaw limited to one letter, with no guarantee the new dataset is cleaner.
  - Live demo words avoid T (the roadmap's KITTI needs replacing) so the demo doesn't depend on the non-standard letter.
- **K and P variants don't matter in practice:** the MLP accepts the standard versions (80% / 100%), so the dataset's slightly different K and P still generalize to the standard signs. Not a reason to exclude them.
- n = 5 per letter: rough, but 0/5 vs 5/5 is a clear signal.

## 6. Live app

### Confidence gate (`src/gate.py`, checks in `scripts/check_gate.py`)

The MLP predicts every frame (~30/s). While the hand moves between letters it flickers through wrong guesses, so raw predictions can't be typed directly. The gate decides when a prediction becomes a typed letter:

1. **Type:** the same letter wins **10 frames in a row** (~⅓ s) with **confidence ≥ 0.8**.
2. **No repeats while held:** after typing, the same letter can't be typed again until another letter is typed or the hand has been gone for **5 frames**. Shorter hand losses (≤ 4 frames) and brief confidence dips don't re-arm, so tracking hiccups don't cause duplicate letters.
3. **Double letters (LL, SS):** while the same letter is held, sliding the wrist sideways by **≥ 0.6 palm lengths** from where it was typed types it again (how signers mark doubles). Measured in palm lengths, not pixels, so it works at any distance from the camera.

- Same gate code is used by the live demo and (later) the web app.
- Checked on 11 fake frame sequences, all pass: one letter per hold, flicker ignored, too-short or low-confidence holds ignored, hand leaving and returning, short hand loss, confidence blip, double via slide, small drift is not a double, "HELLO".
- Settings are first guesses; to tune on the live demo (too slow → fewer frames / lower threshold; flickery → more frames).
- Known limitation: fast signing can skip letters (each needs ~⅓ s stable); a sequence model (CTC) would handle continuous spelling properly.

### "Other" class: hands that aren't signing

**Problem (found in the live demo):** a resting open hand was typed as F. The MLP is a closed-set classifier: it must pick one of the 24 letters for any hand. An open hand has 3–4 spread fingers, closest to F. A confidence threshold can't fix it (the wrong guesses are confident), and a distance-to-letter check doesn't work either: letters vary more within themselves (95% of training hands within 2.7 of their own letter's centroid) than letter centroids differ from each other (~1.7).

**Fix:** a 25th class, "other".
- Data: my own recordings of non-letter poses (`scripts/record_other.py`): open/relaxed/half-closed hands, waving, turning, claw, reaching, hand partly out of frame; both hands; captured automatically ~5×/s. `other-1` = 601 samples (train/val, split in blocks of 50 consecutive captures), `other-test` = 201 samples recorded separately (held-out test). Training data only; never mixed into the letter test sets.
- Retrained the MLP once with the final settings (`--model mlp_other`); everything else unchanged. 52,505 parameters (+129 for the extra output). Saved as `models/mlp_other_final.pt`; the 24-class `mlp_final.pt` stays as the "before".
- The gate treats "other" like an unsure frame: never typed, no streak.

**First result: F swallowed.** The new model called **all 9 of my webcam F's "other"**. Cause: every "other" sample comes from my webcam (camera mounted high, looking down), while every letter sample comes from the dataset (camera level). From above, my F (three spread fingers) looks more like my open hands than like the dataset's F. Same angle confound as Q → P.
- Tried: removing F-like "other" samples (thumb tip within 0.4 palm lengths of the index tip, ~5% of other-1) and retraining. Barely helped (8 of 9 F's still "other"), so bad samples weren't the cause.
- **Pinch guard** (`src/predict.py`): if the thumb tip touches the index tip (< 0.35 palm lengths; 95% of dataset F's are < 0.31, non-letter hands are mostly > 0.4), "other" is ruled out. A thumb–index circle (F, O, D, ...) is a deliberate shape, not a resting hand. Kept the cleaned retrain as well.

**Before / after** (`scripts/compare_other.py`, `other_class_comparison.csv`). "Would type" = a letter predicted with confidence ≥ 0.8, enough for the gate.

| Model | Dataset letter F1 | Webcam letter F1 | Real letters called "other" (webcam) | Non-letter hands recognized | Non-letter hands that would type |
|---|---|---|---|---|---|
| MLP, 24 classes | 98.1% | **93.1%** | 0% | 0% | **69.2%** (mostly F) |
| MLP + other | 98.0% | 87.7% | 3.7% (all F) | 85.1% | 7.5% |
| **MLP + other + pinch guard** | **98.0%** | **91.0%** | **0%** | **84.6%** | **7.5%** |

- **Resting hands typing a letter: 69% → 7.5%** of frames, for −2.1 points of webcam letter F1. A clear win for a practice tool: random F's while resting are worse than slightly lower accuracy, and the gate needs 10 frames in a row anyway.
- Dataset letter F1 is unchanged (98.1 → 98.0).
- Limitation: "other" data is one person, one webcam angle; the guard is a hand-written rule built from one observed failure. A cleaner fix: non-letter examples from the same source and angle as the letters (more signers, varied cameras).
- The app and live demo use `mlp_other_final.pt` + the guard (`letter_probs`). Report numbers for the MLP-vs-CNN comparison stay with the 24-class model.

### Live demo and speed (`scripts/live_demo.py`)

Webcam → MediaPipe → `normalize` → MLP → gate → typed text, with per-frame timings shown on screen and a median / 95th-percentile summary printed on exit.

- **Speed** (1,342 frames, CPU, `mlp_other` + pinch guard):

  | Step | Median | 95th percentile |
  |---|---|---|
  | MediaPipe hand landmarker | 10.58 ms | 13.17 ms |
  | Normalize + MLP + guard | **0.25 ms** | 0.40 ms |
  | **Total processing per frame** | **10.84 ms** | **13.18 ms** |

  - **Target ≤ 33 ms per frame: met, ~3× under.** Processing could sustain ~90 FPS.
  - The live view ran at ~16 FPS: that's the webcam's frame rate (a new frame every ~62 ms; the pipeline finishes in ~11 ms and waits).
  - Raw camera test (no MediaPipe, no model), same room light: DirectShow delivers **~12–13 fps** (ignores MJPG and 30 fps requests); Media Foundation reports **29 fps** with no failed reads over 20 s, but **261 of 582 frames were exact duplicates** of the previous one, i.e. **~16 new images per second** padded to 30. So the camera itself produces ~15–16 frames/s in this light; webcams lengthen exposure in dim light. Kept DirectShow (stable, small difference). Brighter light is the fix, and also helps accuracy (room light 98.6% vs dark 87%).
  - MediaPipe is ~98% of the per-frame cost; the MLP is 0.25 ms. So the CNN's 1.4 ms (section 5) wouldn't have been a speed problem either: the landmark approach wins on accuracy, not speed.
### API skeleton (`app/main.py`, check in `scripts/check_api.py`)

FastAPI server for the web app: `POST /predict` takes the 21 MediaPipe landmarks + video width/height and returns the letter (or "other"), its confidence and the top 3; `GET /health`. Uses `mlp_other_final.pt` + pinch guard through the same `letter_probs` as the live demo. Stateless: the gate will run per user in the app (decided in Phase 4).

- Correctness: 72/72 `room-right` samples give the same answer through the API as calling the model directly (70/72 correct). A malformed request (5 points instead of 21) is rejected with HTTP 422.
- Speed: ~5 ms per request in-process (validation + JSON + model); **2.4 ms round trip over real HTTP** on `127.0.0.1`.
- **Gotcha:** the same request through `localhost` took **~2,040 ms**: on this Windows machine `localhost` tries IPv6 first and waits ~2 s before falling back to IPv4. Always use `http://127.0.0.1:8000`.

### Frontend decision: React (final; changed from the proposal's plain HTML/JS)

| | Plain HTML/JS | React |
|---|---|---|
| Setup | none, FastAPI serves files | Node, npm, Vite, build step |
| Daily dev | one server | Vite + FastAPI (proxy) |
| Demo day | what's edited is what runs | must rebuild before demoing |
| Live UI state (progress bar, guess, word, pass/fail) | manual DOM updates | state → UI automatically |
| Reuse across 3 modes | shared modules, needs discipline | components (`<Camera>`) |
| My familiarity | less practiced | know it |

- **Chosen: React.** The two deciding factors were how much live state the pages have (camera-driven updates 16–30×/s plus mode logic) and familiarity. Setup costs are one-time. Rubric grades "a working web app", not the framework. The proposal said plain HTML/JS; this is a deliberate change.
- Setup: `web/` = Vite + React, MediaPipe via `@mediapipe/tasks-vision`. In development Vite proxies API calls to FastAPI on 127.0.0.1:8000; for the demo, `npm run build` once and FastAPI serves `web/dist`, so it's one command. **Demo checklist: rebuild and test the built version, not only the dev server.**

### Web app setup (Phase 4 start)

- `web/`: Vite 8 + React 19, MediaPipe `@mediapipe/tasks-vision` **1.1.1, the same version as the Python `mediapipe==1.1.1`**, loading the same `hand_landmarker.task` file. `npm install` copies the MediaPipe WebAssembly and the model into `web/public/mediapipe/`, so the app runs offline (no CDN on demo day).
- API routes moved under `/api/` (`/api/predict`, `/api/health`) so pages and API can't clash; Vite proxies `/api` to FastAPI in development; FastAPI serves the built app (`web/dist`) for the demo. Built bundle: 380 KB JS (116 KB gzipped).
- `<Camera>` component shared by all modes: webcam 640×480 → MediaPipe in VIDEO mode on the **raw (unmirrored) frame, like Python**; only the display is mirrored.

### Browser vs Python check (Phase 4 step 1)

- Check page: live prediction from the browser's landmarks, and a "Compare with Python" button that sends the same frame as a JPEG to `/api/debug/compare`, where Python runs its own MediaPipe on it and returns both predictions plus the landmark difference.
- Endpoint verified with simulated browser frames (dataset photos placed in a 640×480 frame): same letter, landmarks **0.3–0.5 px apart** (just JPEG compression). That's the baseline for the real browser test.
- **Real browser test (Chrome, my webcam): 26/26 same letter**, covering all 24 letters (`browser_vs_python.csv`, transcribed from the check page).
  - Landmark difference: **median 3.0 px** (0.46% of the 640 px width), mean 4.0 px, max 12.7 px.
  - Largest gaps (> 5 px): Q 12.7, P 10.8, H 8.5, Y 6.1, N 5.8, A 5.5. P and Q point down and H points sideways, where fingers hide each other most. Likely cause: the browser runs MediaPipe in VIDEO mode (tracks the hand from the previous frame) while the Python check runs IMAGE mode (fresh detection on one frame), and hidden joints are where the two guesses can differ.
  - Even the largest gaps never changed the letter. **No mirroring or coordinate mismatch between browser and Python**, so the browser can feed the same model and normalization as the Python pipeline.
- The camera first failed with "Could not start video source"; it worked after a retry. `<Camera>` now falls back to default camera settings if the preferred 640×480 @ 30 fps request is refused.

### Spell mode (Phase 4, first practice mode)

Design choice: a **goal to follow, not a race**. No timer, no speed score: the app shows a word, the learner spells it letter by letter. (Considered a Monkeytype-style speed game; rejected because the point is learning correct handshapes, and fingerspelling through the gate is inherently ~0.6 s per letter.)

- Current letter highlighted, with its **reference photo**; correct sign → green, moves on; wrong sign → "Looks like X, try again", stays on the letter (a learner can't skip a shape they got wrong). Finished word → "Done! 🎉", next word after 1.2 s.
- **Words / Letters** setting: "Letters" serves single random letters, so the plain drill is the same page.
- Double letters: when the next letter repeats, the panel says "Same letter again: slide your hand sideways a little" (the gate's double-letter rule).
- **Word list** (`web/src/lib/words.js`, 50 words): no T (non-standard in the dataset), J, Z (motion letters), and no word mixing P and Q (the webcam's most confused pair); checked in code. 13 words contain double letters (HELLO, BOOK, COFFEE, ...). T still appears in Letters mode, with the ⚠️ note from the T decision.
- **Reference photos** (`web/public/letters/`, 24 × 240 px, 207 KB): one training photo per letter, chosen automatically as the most typical hand shape (closest to the letter's median normalized landmarks) among that letter's best-lit third (brightness of the hand region, not the background; a first version picked a dark, backlit V). Cropped around the hand and flipped horizontally to match the mirrored camera view.
- **Gate in the browser** (`web/src/lib/gate.js`): JavaScript port of `src/gate.py`, same settings; `web/scripts/check-gate.mjs` runs the same 14 cases as the Python check, all pass.
- Verified in a browser (without camera): layout, word display, reference photo loading, Words/Letters toggle, Skip, T warning. **Not yet verified: signing with the live camera.**

### Learning hub plan

Scope agreed for the app (built in this order; 1–4 core, 5 needed for the business case, 6–7 cut first):
1. Attempt log (SQLite) 2. Categories + home page (spell page becomes "Practice") 3. Exam mode 4. Learner dashboard (letter accuracy grid, exam history) 5. Trainer dashboard + simulated 20-learner class 6. Learn grid (letter photos + tips) 7. Free mode with auto-space.
Deliberately out of scope: login/accounts (one learner per browser, as in the roadmap), videos or long lesson text, levels/XP/achievements.

### Attempt log (`app/db.py`, checks in `scripts/check_db.py`)

SQLite file `data/attempts.db` (gitignored: personal usage data). Two tables:
- `sessions`: one practice run or exam (mode, category, start/end, items in the exam, exam score).
- `attempts`: one row per letter the gate types: item (word), position, target letter, typed letter, correct, confidence, **ms since the letter was asked for**.

API: `POST /api/sessions`, `POST /api/sessions/{id}/finish`, `POST /api/attempts`, `GET /api/progress` (per-letter attempts/accuracy/avg time, top confusions, per-category completed words and best exam, exam history, totals).
- **Exam score = share of letters right on the first try** (later attempts at the same letter don't count).
- A category's word counts as completed when its last letter was signed correctly.
- Checks (throwaway database, real one untouched): 9/9 pass, covering practice and exam sessions, first-try exam scoring (MILK with one miss → 0.75), per-letter accuracy and timing, confusions, category stats, exam history, invalid input rejected.
- The spell page starts a practice session per mode (words / alphabet) and logs every typed letter fire-and-forget (a failed log write never interrupts practice). Verified in a browser: sessions are created on load and on mode switch.

### Categories + home page (hub step 2)

- 7 categories (`web/src/lib/words.js`), 133 items: **Alphabet** (24 letters, includes T with ⚠️), **Everyday** 21, **Food** 21, **Animals** 21, **Colors** 12, **Nature** 16, **Double letters** 18 (every word has a repeated letter, to practise the sideways slide).
- `web/scripts/check-words.mjs` validates every list: only supported letters (no J/Z), no T outside Alphabet, no word mixing P and Q, no duplicates, and every "Double letters" word really has a double. All pass.
- Home page: one card per category with its progress from the attempt log (words completed / total, best exam) and Practice / Exam buttons (Exam arrives in step 3).
- The spell page became **Practice** for a chosen category: same behaviour, the logged session carries the category id. Words/Letters toggle removed (Alphabet is now its own category).
- Verified in a browser: home renders all 7 cards, Food → Practice shows a Food word with the right photo, back returns home, no console errors. Not verified: signing with the live camera.

### Exam mode (hub step 3)

- Start screen explains the rules; the learner presses Start once the skeleton shows on their hand.
- **5 words** from the category (**10 letters** for Alphabet), **no reference photo**. The live guess and progress bar stay visible (they show what the learner is signing, not the answer).
- Each letter ends as **first try / after retries / failed**. A wrong sign shows "Not quite (looked like X)"; after **3 misses** the letter is marked wrong and the exam moves on. "I don't know" fails the letter immediately and shows the answer.
- **Score = letters right on the first try**, computed by the server from the attempt log (see attempt log). "I don't know" is logged with typed = "?": counts as wrong for the score and the letter's accuracy, excluded from the confusion list (it isn't something the model saw). Covered by a new check in `scripts/check_db.py` (10/10 pass).
- Results screen: score, every word with letters coloured by outcome, "letters to practise", buttons for Try again / Practice this category / Categories. The category card shows the best exam score.
- Practice and Exam share `WordDisplay` and `LiveCamera` components.
- Verified in a browser by running a full Colors exam answering "I don't know" to all 24 letters: intro → 5 words with no photo → results 0%, all letters failed, letters-to-practise list, exam saved on the server, card shows "best exam 0%". The test exam was then deleted from `data/attempts.db`. Not verified: an exam signed with the live camera.

### Learner dashboard: "My progress" (hub step 4)

- Stat tiles: letters signed, % signed correctly, exams taken, best exam.
- **Accuracy by letter**: a 24-tile grid (A–Y), each tile printing its accuracy, coloured in 5 bands (< 50%, 50–70, 70–85, 85–95, 95%+); untried letters are outlined. Hover shows attempts, accuracy and average time when right; a Grid/Table toggle shows the same numbers as a table.
  - Colour choice: accuracy is a magnitude, so **one hue (blue) stepped dim → bright**, not red → green (red/green are status colours and the hardest pair for colour-blind viewers). Checked with a palette validator against the app's dark panel: single hue, monotone lightness, visible step gaps. The first ramp failed (darkest step 1.8:1 against the background, below the 2:1 floor), so it was shifted one step lighter (2.2:1). Colour is never the only cue: every tile prints its percentage.
- **Letters to work on**: lowest accuracy below 85%, only letters with ≥ 3 tries (one or two tries is too little to call a letter weak). **Most often mixed up**: "Aiming for M, the model saw N ×3".
- **Exam history**: date, category, score.
- Empty state links to Alphabet practice.
- Verified in a browser with temporary sample data (227 attempts, 4 sessions; removed afterwards): tiles coloured by the right band, untried letter outlined, legend, hover tooltip, table view, focus list, mix-ups, exam history, no horizontal overflow. Not checked by eye (screenshots unavailable at the time).

### Live demo notes

- Gate settings: default (10 frames, 0.8) felt fine in the first live try; note that the 10-frame hold is fps-dependent (~0.6 s at 16 FPS, ~0.33 s at 30 FPS).
- Double-letter rule (tested live):
- Word spaces: ASL marks word breaks with a pause, not a sign. Not needed for the drill and spell-the-word modes; an automatic space after ~1 s without a letter would cover free-spell mode.

## 7. Problems and fixes

- **Webcam stopped sending frames on Windows.** OpenCV's default backend (MSMF) dropped the stream after a few seconds. Fixed by opening the camera with DirectShow (`cv2.CAP_DSHOW`).
- **Webcam freezes during long recording sessions.** In `record_test.py` the preview froze after a few minutes, then eventually "Camera stopped sending frames". Cause: the camera kept returning the same stale frame, which looks like a normal successful read. Fixes in `CameraReader`: retry failed reads for ~1.5 s; treat ~1.5 s of byte-identical frames as frozen (a live sensor always adds noise) and reconnect the camera automatically; request MJPG at 640×480 to reduce USB bandwidth. Tested with a fake camera that freezes; not yet confirmed on the real webcam.

## 8. Report lines
