"""Check the attempt log API against a throwaway database (the real data/attempts.db is untouched).

Run from the project root:  python -m scripts.check_db
"""

import os
import tempfile

TMP = tempfile.mkdtemp()
os.environ["ASL_DB_PATH"] = os.path.join(TMP, "test.db")  # must be set before the app is imported

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def check(name, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return ok


def main():
    c = TestClient(app)
    results = []

    # Practice: spell "GOOD", one miss on the first O (seen as C), slide for the double O
    p = c.post("/api/sessions", json={"mode": "practice", "category": "everyday"}).json()["session_id"]
    for pos, target, typed, ms in [(0, "G", "G", 900), (1, "O", "C", 700), (1, "O", "O", 1500),
                                   (2, "O", "O", 800), (3, "D", "D", 600)]:
        c.post("/api/attempts", json={"session_id": p, "item": "GOOD", "position": pos, "target": target,
                                      "typed": typed, "confidence": 0.9, "ms": ms})
    results.append(check("practice finish has no score", c.post(f"/api/sessions/{p}/finish").json()["score"] is None))

    # Exam: "MILK" with one wrong first try on M -> 3 of 4 letters right first time = 0.75
    e = c.post("/api/sessions", json={"mode": "exam", "category": "food", "items_total": 1}).json()["session_id"]
    for pos, target, typed in [(0, "M", "N"), (0, "M", "M"), (1, "I", "I"), (2, "L", "L"), (3, "K", "K")]:
        c.post("/api/attempts", json={"session_id": e, "item": "MILK", "position": pos, "target": target, "typed": typed})
    score = c.post(f"/api/sessions/{e}/finish").json()["score"]
    results.append(check(f"exam score = first-try accuracy (expected 0.75, got {score})", abs(score - 0.75) < 1e-9))

    prog = c.get("/api/progress").json()
    letters = {r["letter"]: r for r in prog["letters"]}
    results.append(check(f"totals: 10 attempts, 2 sessions ({prog['totals']})",
                         prog["totals"]["attempts"] == 10 and prog["totals"]["sessions"] == 2))
    results.append(check(f"O accuracy 2/3 ({letters['O']['accuracy']:.3f})", abs(letters["O"]["accuracy"] - 2 / 3) < 1e-9))
    results.append(check(f"O avg time when correct 1150 ms ({letters['O']['avg_ms_correct']})",
                         letters["O"]["avg_ms_correct"] == 1150))
    results.append(check(f"top confusions include O->C and M->N ({prog['confusions']})",
                         {(x["target"], x["typed"]) for x in prog["confusions"]} == {("O", "C"), ("M", "N")}))
    cats = {r["category"]: r for r in prog["categories"]}
    results.append(check(f"categories: everyday 1 completed, food best exam 0.75 ({cats})",
                         cats["everyday"]["items_completed"] == 1 and cats["food"]["best_exam"] == 0.75))
    results.append(check("exam history has the exam", [x["id"] for x in prog["exams"]] == [e]))

    bad = c.post("/api/sessions", json={"mode": "quiz", "category": "x"}).status_code
    results.append(check(f"invalid mode rejected (HTTP {bad})", bad == 422))
    print("All passed." if all(results) else "Some failed.")


if __name__ == "__main__":
    main()
