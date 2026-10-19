"""Simulate a class of 20 learners using the app for 4 weeks (SIMULATED, not real users).

Grounded in measurements where possible:
  - p_model[letter]: how often the app accepts a correctly signed letter, measured on my
    webcam test sessions with the deployed model (mlp_other + pinch guard), smoothed.
Stated assumptions where not (all in ASSUMPTIONS below, written into the output):
  - how hard each handshape is for a beginner, learner skill spread, practice frequency,
    learning rate, time per attempt.

Each attempt: the learner forms the right shape with probability p_shape (rises with practice),
and the app accepts it with probability p_model. A letter is retried until accepted (practice)
or up to 3 misses (exam, as in the app). Output: results/simulated_cohort.json

Run from the project root:  python -m scripts.simulate_cohort
"""

import json
import math
from collections import defaultdict

import numpy as np

from src import config
from src.evaluate_webcam import STANDARD_PREFIX, load_final, load_session
from src.predict import letter_probs

OUT = config.RESULTS_DIR / "simulated_cohort.json"

ASSUMPTIONS = {
    "learners": 20,
    "weeks": 4,
    "seed": 42,
    "sessions_per_week_range": [1, 5],       # each learner's engagement, uniform
    "letters_per_session_range": [150, 300], # x 3 s per attempt = 7.5-15 minutes of practice
    "seconds_per_attempt": 3.0,              # sign + gate (~0.6 s steady) + reading feedback
    "skill_sd_logit": 0.7,                   # spread of learner ability
    "learning_per_attempt_logit": 0.12,      # each attempt at a letter makes the next one likelier
    "practice_max_tries": 6,                 # then the learner skips the letter
    "exam_max_misses": 3,                    # same rule as the app
    "exam_words": 5,
    "mastery": "4 of the last 5 attempts at the letter correct",
    # A single "completion" threshold was dropped: with 22/24 it was 0%, because Q and O are often
    # rejected by the app and X/Q barely appear in the word lists, which says more about the app
    # than the learners. Reported instead: letters mastered per learner, and per letter the app's
    # acceptance and practice exposure, so a stuck letter's cause is visible.
    # Beginner's first-try chance of forming the shape right, by handshape group (assumption):
    "shape_start_prob": {"easy": 0.75, "medium": 0.6, "hard": 0.45},
    "shape_groups": {
        "easy": "BCDILOY",      # open, distinct shapes
        "medium": "FKRUVWX",    # finger combinations
        "hard": "AEMNSTGHPQ",   # closed fists (thumb position) and sideways / downward letters
    },
}
WORD_CATEGORIES = ["everyday", "food", "animals", "colors", "nature", "doubles"]


def logit(p):
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def measured_model_acceptance():
    """Per letter: share of my correctly signed webcam samples the deployed app accepts (Laplace-smoothed)."""
    model = load_final("mlp_other")
    hits, totals = defaultdict(int), defaultdict(int)
    sessions = [p.stem for p in config.WEBCAM_TEST_DIR.glob("*.npz") if not p.stem.startswith(STANDARD_PREFIX)]
    for s in sessions:
        with np.load(config.WEBCAM_TEST_DIR / f"{s}.npz") as d:
            for lm, label, (w, h) in zip(d["landmarks"], d["labels"], d["frame_size"]):
                pred = int(letter_probs(model, lm, aspect=w / h).argmax())
                letter = str(label)
                totals[letter] += 1
                hits[letter] += pred < config.NUM_CLASSES and config.LETTERS[pred] == letter
    return {l: (hits[l] + 1) / (totals[l] + 2) for l in config.LETTERS}, sum(totals.values()), sessions


def load_words():
    """The app's word lists, parsed from web/src/lib/words.js so both stay in sync."""
    text = (config.ROOT / "web" / "src" / "lib" / "words.js").read_text(encoding="utf-8")
    words = {}
    for cat in WORD_CATEGORIES:
        block = text.split(f'id: "{cat}"', 1)[1].split("items: [", 1)[1].split("]", 1)[0]
        words[cat] = [w.strip().strip('"') for w in block.replace("\n", " ").split(",") if w.strip()]
    return words


class Learner:
    def __init__(self, idx, rng, start_logit):
        self.name = f"Learner {idx:02d}"
        self.skill = rng.normal(0, ASSUMPTIONS["skill_sd_logit"])
        self.sessions_per_week = rng.uniform(*ASSUMPTIONS["sessions_per_week_range"])
        self.start = start_logit
        self.practice = defaultdict(int)
        self.recent = defaultdict(list)  # last outcomes per letter
        self.rng = rng

    def p_shape(self, letter):
        a = ASSUMPTIONS
        return sigmoid(self.start[letter] + self.skill + a["learning_per_attempt_logit"] * self.practice[letter])

    def attempt(self, letter, p_model):
        ok = self.rng.random() < self.p_shape(letter) * p_model[letter]
        self.practice[letter] += 1
        self.recent[letter] = (self.recent[letter] + [ok])[-5:]
        return ok

    def mastered(self):
        return [l for l in config.LETTERS if len(self.recent[l]) >= 5 and sum(self.recent[l]) >= 4]


def simulate(p_model, words):
    a = ASSUMPTIONS
    rng = np.random.default_rng(a["seed"])
    start = {}
    for group, letters in a["shape_groups"].items():
        for l in letters:
            start[l] = logit(a["shape_start_prob"][group])
    learners = [Learner(i + 1, rng, start) for i in range(a["learners"])]

    letter_stats = defaultdict(lambda: {"attempts": 0, "first_try": 0, "first_try_ok": 0})
    out = []
    for lr in learners:
        weekly = []
        total_attempts = 0
        for week in range(1, a["weeks"] + 1):
            n_sessions = rng.poisson(lr.sessions_per_week)
            for _ in range(n_sessions):
                budget = rng.integers(*a["letters_per_session_range"], endpoint=True)
                while budget > 0:
                    # beginners start on the alphabet, then move to words
                    pool = config.LETTERS if week == 1 and rng.random() < 0.6 else \
                        words[WORD_CATEGORIES[rng.integers(len(WORD_CATEGORIES))]]
                    item = pool[rng.integers(len(pool))]
                    for letter in item:
                        for k in range(a["practice_max_tries"]):
                            ok = lr.attempt(letter, p_model)
                            st = letter_stats[letter]
                            st["attempts"] += 1
                            if k == 0:
                                st["first_try"] += 1
                                st["first_try_ok"] += ok
                            budget -= 1
                            total_attempts += 1
                            if ok:
                                break
            # weekly exam: 5 words, first-try score, up to 3 misses per letter
            cat = WORD_CATEGORIES[rng.integers(len(WORD_CATEGORIES))]
            exam_words = rng.choice(words[cat], size=a["exam_words"], replace=False)
            first = total = 0
            for w in exam_words:
                for letter in w:
                    total += 1
                    for k in range(a["exam_max_misses"]):
                        ok = lr.attempt(letter, p_model)
                        total_attempts += 1
                        if k == 0:
                            first += ok
                        if ok:
                            break
            weekly.append({"week": week, "sessions": int(n_sessions), "letters_mastered": len(lr.mastered()),
                           "exam_category": cat, "exam_score": round(first / total, 3)})
        out.append({
            "name": lr.name,
            "sessions_per_week": round(lr.sessions_per_week, 1),
            "attempts": total_attempts,
            "practice_minutes": round(total_attempts * a["seconds_per_attempt"] / 60, 1),
            "weekly": weekly,
            "letters_mastered": len(lr.mastered()),
            "not_mastered": [l for l in config.LETTERS if l not in lr.mastered()],
        })

    letters = []
    for l in config.LETTERS:
        st = letter_stats[l]
        letters.append({
            "letter": l,
            "attempts": st["attempts"],
            "first_try_accuracy": round(st["first_try_ok"] / st["first_try"], 3) if st["first_try"] else None,
            "attempts_per_success_estimate": round(st["attempts"] / max(st["first_try"], 1), 2),
            "attempts_per_learner": round(st["attempts"] / len(out), 1),
            "model_acceptance": round(p_model[l], 3),
            "learners_mastered": sum(l not in x["not_mastered"] for x in out),
        })
    return out, letters


def main():
    p_model, n_samples, sessions = measured_model_acceptance()
    words = load_words()
    learners, letters = simulate(p_model, words)
    a = ASSUMPTIONS
    final_exams = [x["weekly"][-1]["exam_score"] for x in learners]
    summary = {
        "learners": len(learners),
        "letters_mastered_min_median_max": [int(min(m := [x["letters_mastered"] for x in learners])),
                                            int(np.median(m)), int(max(m))],
        "median_final_exam": round(float(np.median(final_exams)), 3),
        "mean_letters_mastered": round(float(np.mean([x["letters_mastered"] for x in learners])), 1),
        "total_practice_hours": round(sum(x["practice_minutes"] for x in learners) / 60, 1),
        "weekly_mastered_mean": [round(float(np.mean([x["weekly"][w]["letters_mastered"] for x in learners])), 1)
                                 for w in range(a["weeks"])],
        "weekly_exam_mean": [round(float(np.mean([x["weekly"][w]["exam_score"] for x in learners])), 3)
                             for w in range(a["weeks"])],
    }
    result = {
        "simulated": True,
        "note": "SIMULATED cohort for illustrating the trainer dashboard. Not real users.",
        "assumptions": a,
        "model_acceptance": {"source": f"webcam test sessions {sessions} ({n_samples} samples), "
                                       "mlp_other + pinch guard, Laplace-smoothed (hits+1)/(n+2)",
                             "per_letter": {l: round(p, 3) for l, p in p_model.items()}},
        "summary": summary,
        "letters": letters,
        "learners": learners,
    }
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"Saved {OUT.relative_to(config.ROOT)}")
    print(json.dumps(summary, indent=1))
    hardest = sorted(letters, key=lambda x: x["first_try_accuracy"])[:6]
    print("hardest:", ", ".join(f"{x['letter']} {x['first_try_accuracy']:.0%}" for x in hardest))
    print("lowest model acceptance:", sorted(p_model.items(), key=lambda kv: kv[1])[:5])


if __name__ == "__main__":
    main()
