# Final report outline (max 5 pages)

Required by the brief: a **"Model Tuning & Optimization"** section (optimizers tried, learning rate, regularization, before/after tuning). Graded on linking the business problem to the technical solution. Every number below is from `results/RESULTS.md`; write the prose around them. Page budget in brackets.

Suggested title: *Real-time ASL Fingerspelling Practice with Hand Landmarks: an MLP vs CNN Comparison*

---

## 1. Problem and business value [0.5 page]
- Learning fingerspelling needs lots of drill; drill uses instructor time that's better spent on teaching.
- Solution: a practice tool that checks each letter from a webcam in real time. **A training tool, not a translator.**
- Scope: ASL (no public Thai Sign Language dataset), 24 static letters (J, Z need motion).
- Research question: do hand landmarks beat raw images, in accuracy, size and speed, especially on a real webcam?

## 2. Data [0.5 page]
- ASL Alphabet (Kaggle): 3,000 photos × 24 letters = 72,000. Few signers, one room → known to overstate accuracy.
- MediaPipe landmarks: 77.3% found → **87.1%** after retrying failures with gamma brightening (gamma recovered 43.0% of failures, CLAHE 2.1%). Class sizes 1,788 (N) – 2,977 (F).
- Data audit: the dataset's **T is not standard ASL** (correct ASL T rejected 5/5); K and P are variants that still work.
- Figure (optional): `gamma_recovered.png` or `check_A.png`.

## 3. Method [1 page]
- **Pipeline**: webcam → MediaPipe (21 landmarks) → normalization → MLP → confidence gate → typed letter.
- **Normalization**: aspect correction, wrist-relative, scaled by palm length; rotation kept (it distinguishes letters). Nearest-centroid 50.4% → 85.4%. Figure: `normalize_before_after.png`.
- **Split**: block split by recording order (70/15/15); random split leaks (1-NN 98.8% vs 96.4%).
- **Augmentation**: mirrored copies for left hands (train only).
- **Models**: B = MLP 63→256→128→24 (52k params); A = CNN, 3 conv blocks on 64×64 hand crops (97k params), the baseline. Same split, same settings.
- **Gate**: 10 stable frames at ≥ 0.8 confidence; sideways slide for double letters.
- **"Other" class + pinch guard**: stops resting hands typing letters (69% → 7.5% of frames).

## 4. Model Tuning & Optimization [1 page] ← required
- Baseline (first untuned MLP): Adam, lr 1e-3, batch 32 → val F1 96.45%, val loss 0.1175.
- **Optimizers × batch size** (table from RESULTS §4): all within 0.25 F1 points; Adam reaches ~94% by epoch 2–3, SGD+momentum batch 64 needs ~30 epochs for its best. Batch 64 smoother. Figure: `optimizer_comparison.png`.
- **Learning rate schedule**: ReduceLROnPlateau (halve after 2 flat epochs) vs fixed → val loss −11%, late spikes gone. Interaction found: early-stopping patience must exceed the scheduler's.
- **Regularization**: dropout 0.5 vs 0.3 → 0.5 underfits; keep 0.3. Early stopping on val loss.
- **Before → after**: val F1 96.45% → 96.81%; val loss 0.1175 → 0.1013 (−14%). Figure: `schedule_dropout.png`.
- Honest framing: tuning mostly improved **convergence and stability**, not final accuracy; the remaining errors come from the data (M/N, Q/P), not the optimizer.

## 5. Results [1 page]
- **Dataset test** (touched once): MLP 98.1% vs CNN 96.0% macro F1; MLP 0.07 ms vs CNN 1.43 ms per sample, 208 vs 388 KB. Different failure letters (MLP M/N, CNN H↔G). Figure: `confusion_matrices.png`.
- **Real webcam** (216 samples, 3 sessions, one signer): **MLP 93.1% vs CNN 16.0%**. Room light 98.6%, left hand 91.3%, dark 87.0%. Figure: `webcam_vs_dataset.png` (the key figure).
- **Speed**: 10.8 ms processing per frame (target ≤ 33 ms); MediaPipe is ~98% of it.
- **Browser = Python**: 26/26 same letter, landmarks 3 px apart (median).
- Main finding in one sentence: *on the dataset the two models are 2 points apart; on a real camera, ~80. Dataset accuracy alone can't choose the model.*

## 6. Application and business results [0.5 page]
- Web app: Learn, Practice by category, Exam (first-try score), My progress, Trainer dashboard. Every attempt logged (SQLite).
- **Simulated class (label it as simulated)**: 20 learners × 4 weeks; recognition rates measured from the real app, learner behaviour assumed. Median 19/24 letters mastered; exam 77% → 83%. Stuck letters split into **model limits** (O, Q, S) and **content gaps** (T, X barely practised).
- Instructor time: practice hours ÷ group size → 4.4 h (≈ ฿1,756 at ฿400/h, group of 10) for this simulated class. All inputs are assumptions.

## 7. Limitations and future work [0.5 page]
- One signer for testing; few signers in training data.
- Dataset T non-standard → re-record T (several signers) and retrain.
- "Other" class trained on one camera angle; it shifted O → C (found per-letter, hidden by the average F1).
- Camera angle hurts downward letters (Q → P).
- Motion letters (J, Z) and fast continuous spelling need a sequence model (LSTM / CTC); word-level signs need both hands.
- Thai Sign Language once data exists: the pipeline carries over.

## References
- ASL Alphabet dataset (Kaggle) — cite with its license.
- MediaPipe Hand Landmarker (Google).
- An ASL dictionary used to check handshapes (e.g. Lifeprint / Handspeak).
- PyTorch, scikit-learn.

---

### Figures that fit in 5 pages (pick ~5)
1. `normalize_before_after.png` (method)
2. `optimizer_comparison.png` (tuning)
3. `schedule_dropout.png` (tuning)
4. `confusion_matrices.png` (results)
5. `webcam_vs_dataset.png` (results, the headline)

### Things not to forget
- Say "simulated" every time the class data appears.
- State the AI-assistance policy you followed, if the course requires it.
- Check the dataset license before including its photos (app reference images come from it).
