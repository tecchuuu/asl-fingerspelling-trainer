import { useCallback, useEffect, useRef, useState } from "react";
import Camera from "../components/Camera.jsx";
import { logAttempt, startSession } from "../lib/api.js";
import { useLetterStream } from "../lib/useLetterStream.js";
import { LETTER_NOTES, makeQueue } from "../lib/words.js";

const NEXT_DELAY_MS = 1200;

/**
 * Spell mode: a goal word, one letter at a time. The current letter is highlighted with its
 * reference photo; a correct sign turns it green and moves on, a wrong one shows what the
 * model saw and stays. "Letters" setting = single-letter items (the drill).
 */
export default function SpellPage() {
  const [mode, setMode] = useState("words");
  const queue = useRef(makeQueue("words"));
  const [item, setItem] = useState(() => queue.current.pop());
  const [index, setIndex] = useState(0);
  const [feedback, setFeedback] = useState(null); // { kind: "good" | "bad" | "done", text }
  const [session, setSession] = useState({ items: 0, letters: 0, misses: 0 });
  const busy = useRef(false); // true between finishing an item and showing the next
  const sessionId = useRef(null);
  const promptedAt = useRef(performance.now()); // when the current letter was first asked for

  const target = item[index];

  // One logged practice session per mode; a new one when switching Words / Letters
  useEffect(() => {
    sessionId.current = null;
    startSession("practice", mode === "letters" ? "alphabet" : "words")
      .then((id) => (sessionId.current = id))
      .catch((err) => console.warn("session not started:", err));
  }, [mode]);

  useEffect(() => {
    promptedAt.current = performance.now();
  }, [item, index]);

  const nextItem = useCallback((newMode = mode) => {
    if (!queue.current.length) queue.current = makeQueue(newMode);
    setItem(queue.current.pop());
    setIndex(0);
    setFeedback(null);
    busy.current = false;
  }, [mode]);

  const onTyped = useCallback((letter, prediction) => {
    if (busy.current) return;
    if (sessionId.current) {
      logAttempt({
        session_id: sessionId.current, item, position: index, target, typed: letter,
        confidence: prediction?.confidence ?? null,
        ms: Math.round(performance.now() - promptedAt.current),
      });
    }
    if (letter !== target) {
      setFeedback({ kind: "bad", text: `Looks like ${letter}, try again` });
      setSession((s) => ({ ...s, misses: s.misses + 1 }));
      return;
    }
    setSession((s) => ({ ...s, letters: s.letters + 1 }));
    if (index + 1 < item.length) {
      setIndex(index + 1);
      setFeedback({ kind: "good", text: "✓" });
      return;
    }
    busy.current = true;
    setIndex(item.length);
    setFeedback({ kind: "done", text: item.length > 1 ? "Done! 🎉" : "✓ Nice!" });
    setSession((s) => ({ ...s, items: s.items + 1 }));
    setTimeout(() => nextItem(), NEXT_DELAY_MS);
  }, [target, index, item, nextItem]);

  const { onFrame, live, progress, resetGate } = useLetterStream(onTyped);

  function switchMode(newMode) {
    if (newMode === mode) return;
    setMode(newMode);
    queue.current = makeQueue(newMode);
    resetGate();
    nextItem(newMode);
  }

  function skip() {
    resetGate();
    nextItem();
  }

  return (
    <div className="page">
      <div className="toolbar">
        <div className="segmented">
          <button className={mode === "words" ? "on" : ""} onClick={() => switchMode("words")}>Words</button>
          <button className={mode === "letters" ? "on" : ""} onClick={() => switchMode("letters")}>Letters</button>
        </div>
        <span className="muted">
          {session.items} done · {session.letters} letters · {session.misses} misses
        </span>
        <button className="ghost" onClick={skip}>Skip</button>
      </div>

      <div className="word">
        {[...item].map((c, i) => (
          <span key={i} className={i < index ? "done" : i === index ? "current" : ""}>{c}</span>
        ))}
      </div>
      <div className={`feedback ${feedback?.kind ?? ""}`}>{feedback?.text ?? " "}</div>

      <div className="layout">
        <div>
          <Camera onFrame={onFrame} />
          <div className="guess">
            <span>{live ? (live.letter === "other" ? "–" : live.letter) : "no hand"}</span>
            <div className="bar"><div style={{ width: `${progress * 100}%` }} /></div>
          </div>
        </div>

        {target && (
          <div className="panel reference">
            <div className="muted">Sign</div>
            <div className="ref-letter">{target}</div>
            <img src={`/letters/${target}.jpg`} alt={`ASL ${target}`} />
            {LETTER_NOTES[target] && <p className="warn">⚠️ {LETTER_NOTES[target]}</p>}
            {index > 0 && item[index - 1] === target && (
              <p className="hint">Same letter again: slide your hand sideways a little.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
