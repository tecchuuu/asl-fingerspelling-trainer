import { useEffect, useState } from "react";
import { getProgress } from "../lib/api.js";
import { categoryById, LETTERS } from "../lib/words.js";

// Accuracy is a magnitude, so one hue stepped from faint to strong; the steps live in
// styles.css (--acc-1..5) and are validated per theme surface. Red/green stay reserved
// for status; every tile also prints its percentage.
const ACCURACY_BINS = [
  { min: 0, label: "< 50%", step: 1 },
  { min: 0.5, label: "50–70%", step: 2 },
  { min: 0.7, label: "70–85%", step: 3 },
  { min: 0.85, label: "85–95%", step: 4 },
  { min: 0.95, label: "95%+", step: 5 },
];
const FOCUS_MIN_ATTEMPTS = 3; // don't call a letter weak on one or two tries

const binFor = (accuracy) => [...ACCURACY_BINS].reverse().find((b) => accuracy >= b.min);
const pct = (x) => `${Math.round(x * 100)}%`;
const seconds = (ms) => (ms == null ? "–" : `${(ms / 1000).toFixed(1)} s`);

/** Learner dashboard: totals, letter accuracy grid, letters to work on, confusions, exam history. */
export default function ProgressPage({ onStart }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [asTable, setAsTable] = useState(false);
  const [hover, setHover] = useState(null); // { letter, x, y }

  useEffect(() => {
    getProgress().then(setData).catch((err) => setError(err.message));
  }, []);

  if (error) return <div className="page"><p className="bad">Couldn't load progress: {error}</p></div>;
  if (!data) return <div className="page"><p className="muted">Loading...</p></div>;

  const byLetter = Object.fromEntries(data.letters.map((l) => [l.letter, l]));
  const { totals, exams, confusions } = data;
  const bestExam = exams.length ? Math.max(...exams.map((e) => e.score ?? 0)) : null;
  const focus = data.letters
    .filter((l) => l.attempts >= FOCUS_MIN_ATTEMPTS && l.accuracy < 0.85)
    .sort((a, b) => a.accuracy - b.accuracy)
    .slice(0, 5);

  if (totals.attempts === 0) {
    return (
      <div className="page">
        <h1>My progress</h1>
        <div className="panel empty">
          <p>No practice yet. Sign a few letters and your progress shows up here.</p>
          <button onClick={() => onStart("alphabet", "practice")}>Practice the alphabet</button>
        </div>
      </div>
    );
  }

  const hovered = hover && byLetter[hover.letter];

  return (
    <div className="page">
      <h1>My progress</h1>

      <div className="tiles">
        <div className="tile"><div className="tile-value">{totals.attempts}</div><div className="tile-label">letters signed</div></div>
        <div className="tile"><div className="tile-value">{pct(totals.accuracy)}</div><div className="tile-label">signed correctly</div></div>
        <div className="tile"><div className="tile-value">{exams.length}</div><div className="tile-label">exams taken</div></div>
        <div className="tile"><div className="tile-value">{bestExam != null ? pct(bestExam) : "–"}</div><div className="tile-label">best exam</div></div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>Accuracy by letter</h2>
          <div className="segmented">
            <button className={asTable ? "" : "on"} onClick={() => setAsTable(false)}>Grid</button>
            <button className={asTable ? "on" : ""} onClick={() => setAsTable(true)}>Table</button>
          </div>
        </div>

        {!asTable ? (
          <>
            <div className="letter-grid" onMouseLeave={() => setHover(null)}>
              {LETTERS.map((letter) => {
                const s = byLetter[letter];
                const bin = s && binFor(s.accuracy);
                return (
                  <div
                    key={letter}
                    className={`letter-tile ${bin ? `bin-${bin.step}` : "untried"}`}
                    onMouseEnter={(e) => {
                      const r = e.currentTarget.getBoundingClientRect();
                      setHover({ letter, x: r.left + r.width / 2, y: r.top });
                    }}
                  >
                    <span className="tile-letter">{letter}</span>
                    <span className="tile-acc">{s ? pct(s.accuracy) : "–"}</span>
                  </div>
                );
              })}
            </div>
            <div className="legend-row small">
              {ACCURACY_BINS.map((b) => (
                <span key={b.label}><i style={{ background: `var(--acc-${b.step})` }} />{b.label}</span>
              ))}
              <span><i className="untried-swatch" />not tried yet</span>
            </div>
            {hover && (
              <div className="tooltip" style={{ left: hover.x, top: hover.y }}>
                <b>{hover.letter}</b>{" "}
                {hovered
                  ? `: ${pct(hovered.accuracy)} right over ${hovered.attempts} tries, ${seconds(hovered.avg_ms_correct)} on average when right`
                  : ": not tried yet"}
              </div>
            )}
          </>
        ) : (
          <table>
            <thead><tr><th>Letter</th><th>Attempts</th><th>Correct</th><th>Avg time when right</th></tr></thead>
            <tbody>
              {LETTERS.map((letter) => {
                const s = byLetter[letter];
                return (
                  <tr key={letter}>
                    <td><b>{letter}</b></td>
                    <td>{s?.attempts ?? 0}</td>
                    <td>{s ? pct(s.accuracy) : "–"}</td>
                    <td>{seconds(s?.avg_ms_correct)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      <div className="two-col">
        <div className="panel">
          <h2>Letters to work on</h2>
          {focus.length ? (
            <ul className="plain">
              {focus.map((l) => (
                <li key={l.letter}>
                  <b>{l.letter}</b> <span className="muted">{pct(l.accuracy)} correct over {l.attempts} tries</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">Nothing below 85% yet (letters need {FOCUS_MIN_ATTEMPTS}+ tries to count).</p>
          )}
          {confusions.length > 0 && (
            <>
              <h3>Most often mixed up</h3>
              <ul className="plain">
                {confusions.slice(0, 5).map((c) => (
                  <li key={c.target + c.typed}>
                    Aiming for <b>{c.target}</b>, the model saw <b>{c.typed}</b> <span className="muted">×{c.n}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>

        <div className="panel">
          <h2>Exam history</h2>
          {exams.length ? (
            <table>
              <thead><tr><th>Date</th><th>Category</th><th>Score</th></tr></thead>
              <tbody>
                {exams.map((e) => (
                  <tr key={e.id}>
                    <td>{e.started_at.replace("T", " ").slice(0, 16)}</td>
                    <td>{categoryById(e.category)?.name ?? e.category}</td>
                    <td>{e.score != null ? pct(e.score) : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">No exams yet. Try one from a category card.</p>
          )}
        </div>
      </div>
    </div>
  );
}
