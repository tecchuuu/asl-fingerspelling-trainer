"""Start the app's progress from zero, e.g. before a demo. Nothing is lost: the current
attempt log is copied to data/attempts-backup-<date-time>.db first, then emptied in place
(so it works while the server is running).

Run from the project root:
    python -m scripts.reset_progress
Restore a backup: stop the server, copy the backup file over data/attempts.db.
"""

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from app import db


def main():
    db.init()  # make sure the file and tables exist
    path = Path(db.DB_PATH)
    backup = path.with_name(f"attempts-backup-{datetime.now():%Y%m%d-%H%M%S}.db")
    with closing(sqlite3.connect(path)) as conn:
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        attempts = conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        with closing(sqlite3.connect(backup)) as dest:
            conn.backup(dest)  # SQLite's own consistent copy, safe even mid-use
        with conn:
            conn.execute("DELETE FROM attempts")
            conn.execute("DELETE FROM sessions")
            conn.execute("DELETE FROM sqlite_sequence")  # ids start from 1 again
    print(f"Backed up {sessions} sessions and {attempts} attempts to {backup.name}")
    print("Progress reset: the app now starts from zero.")


if __name__ == "__main__":
    main()
