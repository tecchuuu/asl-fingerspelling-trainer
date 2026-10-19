import { useState } from "react";
import { LETTER_NOTES, LETTER_TIPS, LETTERS } from "../lib/words.js";

/** Reference for every letter: photo, one-line tip, common mix-ups, and a "Try it" button. */
export default function LearnPage({ onTry }) {
  const [selected, setSelected] = useState("A");
  const i = LETTERS.indexOf(selected);
  const info = LETTER_TIPS[selected];
  const go = (step) => setSelected(LETTERS[(i + step + LETTERS.length) % LETTERS.length]);

  return (
    <div className="page">
      <h1>Learn the alphabet</h1>
      <p className="hint">
        Pick a letter to see its handshape. Photos are mirrored to match your camera view.
        J and Z need motion and aren't covered.
      </p>
      <div className="learn">
        <div className="learn-grid">
          {LETTERS.map((l) => (
            <button key={l} className={`learn-cell ${l === selected ? "on" : ""}`} onClick={() => setSelected(l)}
              aria-label={`Letter ${l}`}>
              <img src={`/letters/${l}.jpg`} alt="" />
              <span>{l}</span>
            </button>
          ))}
        </div>

        <div className="panel learn-detail">
          <div className="learn-head">
            <button className="ghost icon" onClick={() => go(-1)} aria-label="Previous letter">‹</button>
            <div className="ref-letter">{selected}</div>
            <button className="ghost icon" onClick={() => go(1)} aria-label="Next letter">›</button>
          </div>
          <img src={`/letters/${selected}.jpg`} alt={`ASL ${selected}`} />
          <p>{info.tip}</p>
          {info.confusedWith && <p className="muted small">Easily mixed up with: <b>{info.confusedWith}</b></p>}
          {LETTER_NOTES[selected] && <p className="warn">⚠️ {LETTER_NOTES[selected]}</p>}
          <button onClick={() => onTry(selected)}>Try {selected} with the camera</button>
        </div>
      </div>
    </div>
  );
}
