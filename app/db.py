"""Attempt log (SQLite): every letter the gate types, in practice or exam.

Two tables:
  sessions  one practice run or exam (mode, category, start/end, exam score)
  attempts  one typed letter: what was asked, what the model saw, correct or not, how long it took

Progress, exam history and the trainer dashboard are all queries over these.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from src import config

# Overridable so checks can use a throwaway database
DB_PATH = os.environ.get("ASL_DB_PATH", str(config.DATA_DIR / "attempts.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mode        TEXT NOT NULL CHECK (mode IN ('practice', 'exam')),
    category    TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    items_total INTEGER,          -- exam: number of words in the exam
    score       REAL              -- exam: share of letters correct, 0..1
);
CREATE TABLE IF NOT EXISTS attempts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    at          TEXT NOT NULL,
    item        TEXT NOT NULL,    -- the word (or single letter) being spelled
    position    INTEGER NOT NULL, -- index of the letter within the item
    target      TEXT NOT NULL,    -- letter asked for
    typed       TEXT NOT NULL,    -- letter the gate typed
    correct     INTEGER NOT NULL, -- 1 / 0
    confidence  REAL,
    ms          INTEGER           -- time since this letter was asked for
);
CREATE INDEX IF NOT EXISTS attempts_target ON attempts(target);
CREATE INDEX IF NOT EXISTS attempts_session ON attempts(session_id);
"""


def now():
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect():
    # One short connection per request: simplest safe choice with FastAPI's thread pool
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)


def start_session(mode, category, items_total=None):
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (mode, category, started_at, items_total) VALUES (?, ?, ?, ?)",
            (mode, category, now(), items_total),
        )
        return cur.lastrowid


def finish_session(session_id):
    """Close a session; for exams the score is the share of letters right on the first try."""
    with connect() as conn:
        row = conn.execute("SELECT mode FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        score = None
        if row["mode"] == "exam":
            score = conn.execute(FIRST_TRY_SCORE, (session_id,)).fetchone()["score"]
        conn.execute("UPDATE sessions SET finished_at = ?, score = ? WHERE id = ?", (now(), score, session_id))
        return score


# A letter counts as right if its first attempt was correct
FIRST_TRY_SCORE = """
SELECT AVG(correct) AS score FROM (
    SELECT correct FROM attempts a
    WHERE session_id = ?
      AND id = (SELECT MIN(id) FROM attempts b
                WHERE b.session_id = a.session_id AND b.item = a.item AND b.position = a.position)
)
"""


def add_attempt(session_id, item, position, target, typed, confidence=None, ms=None):
    with connect() as conn:
        conn.execute(
            "INSERT INTO attempts (session_id, at, item, position, target, typed, correct, confidence, ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, now(), item, position, target, typed, int(target == typed), confidence, ms),
        )


def progress():
    """Everything the learner dashboard needs, in one call."""
    with connect() as conn:
        letters = conn.execute("""
            SELECT target AS letter,
                   COUNT(*) AS attempts,
                   AVG(correct) AS accuracy,
                   AVG(CASE WHEN correct THEN ms END) AS avg_ms_correct
            FROM attempts GROUP BY target ORDER BY target
        """).fetchall()
        confusions = conn.execute("""
            SELECT target, typed, COUNT(*) AS n FROM attempts
            WHERE correct = 0 GROUP BY target, typed ORDER BY n DESC LIMIT 10
        """).fetchall()
        categories = conn.execute("""
            SELECT s.category,
                   -- an item is completed when its last letter was signed correctly
                   COUNT(DISTINCT CASE WHEN a.correct AND a.position = LENGTH(a.item) - 1
                                       THEN a.item END) AS items_completed,
                   MAX(CASE WHEN s.mode = 'exam' THEN s.score END) AS best_exam
            FROM sessions s LEFT JOIN attempts a ON a.session_id = s.id
            GROUP BY s.category
        """).fetchall()
        exams = conn.execute("""
            SELECT id, category, started_at, finished_at, items_total, score FROM sessions
            WHERE mode = 'exam' AND finished_at IS NOT NULL ORDER BY id DESC LIMIT 20
        """).fetchall()
        totals = conn.execute("""
            SELECT COUNT(*) AS attempts, COALESCE(AVG(correct), 0) AS accuracy,
                   COUNT(DISTINCT session_id) AS sessions FROM attempts
        """).fetchone()
    return {
        "totals": dict(totals),
        "letters": [dict(r) for r in letters],
        "confusions": [dict(r) for r in confusions],
        "categories": [dict(r) for r in categories],
        "exams": [dict(r) for r in exams],
    }
