import { useEffect, useState } from "react";
import { getProgress } from "../lib/api.js";
import { CATEGORIES } from "../lib/words.js";

/** Category cards with progress from the attempt log. */
export default function HomePage({ onStart }) {
  const [progress, setProgress] = useState(null);

  useEffect(() => {
    getProgress().then(setProgress).catch(() => setProgress({ categories: [], totals: null }));
  }, []);

  const statsFor = (id) => progress?.categories.find((c) => c.category === id);

  return (
    <div className="page">
      <h1>Fingerspelling practice</h1>
      <p className="hint">
        Pick a category. <b>Practice</b> shows the reference photo and lets you retry;
        <b> Exam</b> hides the photo and scores your first try.
        {progress?.totals?.attempts > 0 &&
          ` So far: ${progress.totals.attempts} letters signed, ${Math.round(progress.totals.accuracy * 100)}% correct.`}
      </p>

      <div className="cards">
        {CATEGORIES.map((c) => {
          const s = statsFor(c.id);
          const done = s?.items_completed ?? 0;
          return (
            <div key={c.id} className="card">
              <div className="card-head">
                <span className="emoji">{c.emoji}</span>
                <div>
                  <div className="card-title">{c.name}</div>
                  <div className="muted small">{c.items.length} {c.id === "alphabet" ? "letters" : "words"}</div>
                </div>
              </div>
              <p className="muted small card-desc">{c.description ?? c.items.slice(0, 4).join(", ") + ", ..."}</p>
              <div className="bar thin"><div style={{ width: `${(done / c.items.length) * 100}%` }} /></div>
              <div className="muted small card-stats">
                <span>{done}/{c.items.length} done</span>
                <span>{s?.best_exam != null ? `best exam ${Math.round(s.best_exam * 100)}%` : "no exam yet"}</span>
              </div>
              <div className="card-actions">
                <button onClick={() => onStart(c.id, "practice")}>Practice</button>
                <button className="ghost" onClick={() => onStart(c.id, "exam")}>Exam</button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
