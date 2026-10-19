import { useEffect, useState } from "react";
import HandSkeleton from "../components/HandSkeleton.jsx";
import { getProgress } from "../lib/api.js";
import { CATEGORIES } from "../lib/words.js";

const HERO_WORD = "HELLO";

/** Home: the hero spells a word in hand skeletons; below, one row per practice set. */
export default function HomePage({ onStart, onLearn }) {
  const [progress, setProgress] = useState(null);

  useEffect(() => {
    getProgress().then(setProgress).catch(() => setProgress({ categories: [], totals: null }));
  }, []);

  const statsFor = (id) => progress?.categories.find((c) => c.category === id);
  const totals = progress?.totals;

  return (
    <div className="page">
      <section className="hero">
        <div className="hero-word" aria-hidden="true">
          {[...HERO_WORD].map((l, i) => (
            <figure key={i}>
              <HandSkeleton letter={l} size={112} />
              <figcaption>{l}</figcaption>
            </figure>
          ))}
        </div>
        <h1 className="hero-title">Spell it with your hands.</h1>
        <p className="lede">
          Sign each letter to your webcam and see straight away whether the shape is right.
          Practice with hints, then take an exam without them.
        </p>
        <div className="hero-actions">
          <button onClick={() => onStart("alphabet", "practice")}>Start with the alphabet</button>
          <button className="ghost" onClick={onLearn}>See how each letter looks</button>
        </div>
        {totals?.attempts > 0 && (
          <p className="muted small">
            You've signed {totals.attempts} letters so far, {Math.round(totals.accuracy * 100)}% of them right.
          </p>
        )}
      </section>

      <h2 className="section-title">Practice sets</h2>
      <ul className="sets">
        {CATEGORIES.map((c) => {
          const s = statsFor(c.id);
          // Capped: items logged under a category that later left its word list still count in the log
          const done = Math.min(s?.items_completed ?? 0, c.items.length);
          const unit = c.id === "alphabet" ? "letters" : "words";
          return (
            <li key={c.id} className="set">
              <div className="set-name">
                <span className="set-sample" aria-hidden="true">{c.sample}</span>
                <span className="set-title">{c.name}</span>
                <span className="muted small">{c.description ?? `${c.items.length} ${unit}, like ${c.items.slice(0, 3).join(", ")}`}</span>
              </div>
              <div className="set-progress">
                <div className="bar thin" role="progressbar" aria-valuemin={0} aria-valuemax={c.items.length}
                  aria-valuenow={done} aria-label={`${c.name}: ${done} of ${c.items.length} ${unit} done`}>
                  <div style={{ width: `${(done / c.items.length) * 100}%` }} />
                </div>
                <span className="small muted">
                  {done} of {c.items.length} {unit} done
                  {s?.best_exam != null && <>, best exam {Math.round(s.best_exam * 100)}%</>}
                </span>
              </div>
              <div className="set-actions">
                <button onClick={() => onStart(c.id, "practice")}>Practice</button>
                <button className="ghost" onClick={() => onStart(c.id, "exam")}>Exam</button>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
