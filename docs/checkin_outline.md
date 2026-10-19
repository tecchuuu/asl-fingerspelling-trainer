# Check-in outline (Week 12, 10–15 min)

What the brief asks for at the check-in: **data preprocessing done**, **a baseline model that runs with initial results**, **≥ 2 optimizers compared with loss/accuracy**, **initial results and challenges**. Every number below is from `results/RESULTS.md`; figures are in `results/figures/`.

Suggested timing: 9 slides, ~1–1.5 min each, plus a 2–3 min live demo if there's time.

---

## 1. Problem (1 min)
- Fingerspelling is learned by drilling letters with an instructor. The app takes over the **drill**, not the teaching: sign a letter, get instant feedback.
- Scope: ASL, 24 static letters (J, Z need motion). No public Thai Sign Language dataset → TSL is future work.
- Main question: **do hand landmarks beat raw images**, in accuracy, size and speed?

## 2. Data (1.5 min)
- ASL Alphabet (Kaggle): 3,000 photos per letter, 72,000 for 24 letters. `nothing` dropped (no hand → no landmarks; "no hand" plays that role in the app).
- MediaPipe hand landmarks: first pass found a hand in **77.3%** of images. Failures were mostly backlit (dark hand, bright window), clustered in runs of frames.
  - Tried brightening the failures: CLAHE recovered **2.1%**, gamma 0.5 recovered **43.0%** → kept gamma. Final: **87.1%** found (62,704 images). Every sample is flagged if recovered.
  - Figures: `check_A.png` (detected vs failed), `gamma_recovered.png`.
- Data issue found: the dataset's **T isn't standard ASL** (thumb out sideways). Measured: the model rejects a correct ASL T 5/5.

## 3. Preprocessing (1.5 min)
- Landmark normalization: aspect fix (webcam 4:3), subtract the wrist, divide by palm length. **Rotation not normalized on purpose** (P/Q point down, G/H sideways).
- Proof it works: nearest-centroid accuracy **50.4% raw → 85.4% normalized**, no training. Figure: `normalize_before_after.png`.
- **Block split** (frames are numbered in recording order, neighbours are near-duplicates): 1-nearest-neighbour scores 98.8% on a random split vs 96.4% on the block split, evidence that a random split leaks.
- Mirror augmentation (flip x) for left hands.

## 4. Baseline model (1 min)
- MLP: 63 inputs → 256 → 128 → 24, ReLU, dropout 0.3, **52k parameters**.
- First run (Adam, lr 1e-3, batch 32): **val macro F1 96.45%**, 108 s on CPU. This is the "Baseline" in the rubric's sense (the untuned first model).

## 5. Tuning (2 min) ← the rubric's main item
| Optimizer | Batch | Val F1 | Best epoch |
|---|---|---|---|
| Adam | 32 | 96.45% | 9 |
| **Adam** | **64** | **96.60%** | 20 |
| SGD + momentum | 32 | 96.37% | 15 |
| SGD + momentum | 64 | 96.44% | 30 |
- All within 0.25 points: **Adam converges much faster for the same final quality**, not "more accurate". Figure: `optimizer_comparison.png`.
- On the winner: **LR schedule (ReduceLROnPlateau) cut val loss 11%** (0.1133 → 0.1013) and removed late spikes; **dropout 0.5 was worse** (underfits a small net) → kept 0.3. Figure: `schedule_dropout.png`.
- Lesson: early-stopping patience must exceed the scheduler's patience (first attempt stopped one epoch after the LR drop).
- Before → after tuning: val F1 96.45% → **96.81%**, val loss 0.1175 → **0.1013**.

## 6. Results on the dataset (1 min)
- Test set (8,583 samples, touched once): **MLP 98.1%** vs **CNN 96.0%** macro F1. MLP: half the parameters, 20× faster (0.07 vs 1.43 ms).
- They fail on different letters: MLP on M/N (thumb hidden in a fist), CNN on H↔G. Figure: `confusion_matrices.png`.

## 7. Real webcam (1.5 min) ← the headline
- My own webcam test set: 3 sessions (room light right/left hand, dark), 216 samples, recorded before looking at any results.
- **MLP 93.1% vs CNN 16.0%** (CNN at chance level in the dark). On the dataset they were 2 points apart; on a real camera ~80. Figure: `webcam_vs_dataset.png`.
- Why: the CNN learned the look of one signer's room; landmarks carry no appearance.
- MLP: above the 80% webcam target in every session; left hand −7 points, dark −12.

## 8. Challenges (1.5 min)
- **Open hand typed as F**: closed-set classifier. Added an "other" class (my recordings of non-letter hands): resting hands typing a letter 69% → 7.5%. It swallowed my F's (camera-angle confound) → **pinch guard** rule fixed it. Later found it also shifted **O → C** (hidden by the average F1) → kept as a known limitation.
- Webcam freezing / dropping frames on Windows → DirectShow + automatic reconnect.
- `localhost` cost 2 s per request (IPv6 fallback) → 127.0.0.1, 2.4 ms.
- Q ↔ P on the webcam: the camera is mounted high, looking down at pointing-down letters.

## 9. Status and next steps (1 min)
- Done: pipeline, models, tuning, test + webcam evaluation, live demo (10.8 ms per frame, target ≤ 33 ms), web app (practice, exam, progress, trainer dashboard with a simulated class).
- Next: report, demo practice + backup video, optional T fix from another dataset.

---

## Likely questions (one-line answers)
- **Why landmarks, not images?** MediaPipe removes lighting, background and skin before the model sees anything; the CNN baseline measures how much that matters (93% vs 16% on a webcam).
- **Isn't 98% suspicious?** Yes, the dataset is one signer in one room; the block split reduces leakage but can't remove it. That's why the webcam test exists.
- **Why is test higher than val?** The test blocks happen to be easier. It moves both models the same way, so the comparison holds; treat absolute numbers as ±1–2 points.
- **Why Adam?** Same final quality as SGD, reaches it in a fraction of the epochs, smoother with batch 64.
- **Why this learning rate?** 1e-3 is Adam's standard default; the schedule then halves it when val loss stalls.
- **Why dropout 0.3, not 0.5?** 0.5 underfit (train loss rose, val got worse).
- **Why not normalize rotation?** Orientation distinguishes letters (P/Q vs K, G/H vs U).
- **Why 24 classes, not 26?** J and Z need motion; future work with a sequence model.
- **Is the class data real?** No, it's simulated and labelled so everywhere; recognition rates in it are measured from the real app.
