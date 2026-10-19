import { useCallback, useEffect, useRef, useState } from "react";
import LiveCamera from "../components/LiveCamera.jsx";
import WordDisplay from "../components/WordDisplay.jsx";
import { finishSession, logAttempt, startSession } from "../lib/api.js";
import { useLetterStream } from "../lib/useLetterStream.js";
import { categoryById, makeQueue } from "../lib/words.js";

const WORDS_PER_EXAM = 5;
const LETTERS_PER_EXAM = 10; // alphabet category: single letters
const MAX_MISSES = 3;        // after this many wrong signs a letter is marked failed and the exam moves on
const NEXT_DELAY_MS = 900;
const UNKNOWN = "?";         // logged as the typed letter for "I don't know"

/**
 * Exam: a fixed set of items, no reference photo. Score = letters right on the first try
 * (computed by the server from the attempt log). Each letter ends as "first", "retry" or "failed".
 */
export default function ExamPage({ categoryId, onBack, onPractice }) {
  const category = categoryById(categoryId);
  const [phase, setPhase] = useState("intro"); // intro | running | done
  const [items, setItems] = useState([]);
  const [itemIdx, setItemIdx] = useState(0);
  const [index, setIndex] = useState(0);
  const [statuses, setStatuses] = useState([]); // per item, per letter
  const [feedback, setFeedback] = useState(null);
  const [score, setScore] = useState(null);
  const misses = useRef(0);
  const busy = useRef(false);
  const sessionId = useRef(null);
  const promptedAt = useRef(performance.now());

  const item = items[itemIdx] ?? "";
  const target = item[index];

  useEffect(() => {
    promptedAt.current = performance.now();
  }, [itemIdx, index]);

  async function start() {
    const count = category.id === "alphabet" ? LETTERS_PER_EXAM : WORDS_PER_EXAM;
    const chosen = makeQueue(category).slice(0, count);
    setItems(chosen);
    setStatuses(chosen.map((w) => Array(w.length).fill(null)));
    setItemIdx(0);
    setIndex(0);
    setFeedback(null);
    setScore(null);
    misses.current = 0;
    busy.current = false;
    resetGate();
    try {
      sessionId.current = await startSession("exam", category.id, chosen.length);
    } catch (err) {
      sessionId.current = null;
      console.warn("exam session not started:", err);
    }
    setPhase("running");
  }

  const log = useCallback((typed, confidence = null) => {
    if (!sessionId.current) return;
    logAttempt({
      session_id: sessionId.current, item, position: index, target, typed, confidence,
      ms: Math.round(performance.now() - promptedAt.current),
    });
  }, [item, index, target]);

  const finish = useCallback(async () => {
    setPhase("done");
    if (!sessionId.current) return;
    try {
      const r = await finishSession(sessionId.current);
      setScore(r.score);
    } catch (err) {
      console.warn("exam not finished on server:", err);
    }
  }, []);

  // Record this letter's result and move to the next letter / word / results
  const settle = useCallback((status) => {
    setStatuses((prev) => prev.map((row, i) => (i === itemIdx ? row.map((s, j) => (j === index ? status : s)) : row)));
    misses.current = 0;
    if (index + 1 < item.length) {
      setIndex(index + 1);
      return;
    }
    busy.current = true;
    setFeedback({ kind: "done", text: itemIdx + 1 < items.length ? "Next word..." : "Finished!" });
    setTimeout(() => {
      busy.current = false;
      setFeedback(null);
      if (itemIdx + 1 < items.length) {
        setItemIdx(itemIdx + 1);
        setIndex(0);
      } else {
        finish();
      }
    }, NEXT_DELAY_MS);
  }, [index, item, itemIdx, items.length, finish]);

  const onTyped = useCallback((letter, prediction) => {
    if (phase !== "running" || busy.current) return;
    log(letter, prediction?.confidence ?? null);
    if (letter === target) {
      setFeedback(null);
      settle(misses.current === 0 ? "first" : "retry");
      return;
    }
    misses.current += 1;
    if (misses.current >= MAX_MISSES) {
      setFeedback({ kind: "bad", text: `Marked wrong (looked like ${letter})` });
      settle("failed");
    } else {
      setFeedback({ kind: "bad", text: `Not quite (looked like ${letter}). ${MAX_MISSES - misses.current} tries left` });
    }
  }, [phase, target, log, settle]);

  const { onFrame, live, progress, resetGate } = useLetterStream(onTyped);

  function dontKnow() {
    if (busy.current) return;
    log(UNKNOWN);
    setFeedback({ kind: "bad", text: `It was ${target}` });
    settle("failed");
  }

  const flat = statuses.flat();
  const missed = [...new Set(items.flatMap((w, i) => [...w].filter((_, j) => statuses[i]?.[j] && statuses[i][j] !== "first")))];

  return (
    <div className="page">
      <div className="toolbar">
        <button className="ghost" onClick={onBack}>All sets</button>
        <span className="toolbar-title">{category.name} exam</span>
        {phase === "running" && (
          <span className="muted">{category.id === "alphabet" ? "Letter" : "Word"} {itemIdx + 1} of {items.length}</span>
        )}
      </div>

      {phase === "intro" && (
        <div className="panel exam-intro">
          <h2>Ready?</h2>
          <ul>
            <li>{category.id === "alphabet" ? `${LETTERS_PER_EXAM} letters` : `${WORDS_PER_EXAM} words`} from {category.name}</li>
            <li>No reference photo: you have to know the sign</li>
            <li>Your score counts letters right on the <b>first try</b></li>
            <li>After {MAX_MISSES} misses a letter is marked wrong and the exam moves on</li>
          </ul>
          <p className="muted small">Check that the skeleton shows on your hand, then start.</p>
          <button onClick={start}>Start exam</button>
        </div>
      )}

      {phase === "running" && (
        <>
          <WordDisplay item={item} index={index} statuses={statuses[itemIdx]?.map((s, j) => s ?? (j === index ? "current" : null))} />
          <div className={`feedback ${feedback?.kind ?? ""}`}>{feedback?.text ?? " "}</div>
        </>
      )}

      {phase === "done" && (
        <div className="panel exam-results">
          <h2>{score != null ? `${Math.round(score * 100)}%` : `${Math.round((flat.filter((s) => s === "first").length / flat.length) * 100)}%`}</h2>
          <p className="muted">letters right on the first try</p>
          <div className="result-words">
            {items.map((w, i) => <WordDisplay key={i} item={w} index={-1} statuses={statuses[i]} />)}
          </div>
          <p className="legend small">
            <span className="first">■</span> first try <span className="retry">■</span> after retries <span className="failed">■</span> missed
          </p>
          {missed.length > 0
            ? <p>Letters to practise: <b>{missed.join(", ")}</b></p>
            : <p>Perfect: every letter right on the first try 🎉</p>}
          <div className="card-actions">
            <button onClick={start}>Try again</button>
            <button className="ghost" onClick={onPractice}>Practice {category.name}</button>
            <button className="ghost" onClick={onBack}>All sets</button>
          </div>
        </div>
      )}

      {phase !== "done" && (
        <div className="layout">
          <LiveCamera onFrame={onFrame} live={live} progress={progress} />
          {phase === "running" && (
            <div className="panel reference">
              <div className="ref-letter" aria-label={`Sign ${target}`}>{target}</div>
              <p className="muted small">No photo in the exam: sign it from memory.</p>
              {index > 0 && item[index - 1] === target && (
                <p className="hint">Same letter again: slide your hand sideways a little.</p>
              )}
              <button className="ghost" onClick={dontKnow}>I don't know</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
