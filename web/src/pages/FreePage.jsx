import { useCallback, useEffect, useRef, useState } from "react";
import LiveCamera from "../components/LiveCamera.jsx";
import { useLetterStream } from "../lib/useLetterStream.js";

// ASL marks word breaks with a pause, not a sign: a hand that's gone or resting ("other")
// this long after a letter adds a space.
const SPACE_AFTER_MS = 1200;

/** Free spelling: no target, whatever you sign is typed. Not logged (there's no right answer). */
export default function FreePage() {
  const [text, setText] = useState("");
  const [copied, setCopied] = useState(false);
  const spaceTimer = useRef(null);

  const onTyped = useCallback((letter) => setText((t) => t + letter), []);
  const { onFrame, live, progress } = useLetterStream(onTyped);

  const paused = !live || live.letter === "other";
  useEffect(() => {
    clearTimeout(spaceTimer.current);
    if (paused && text && !text.endsWith(" ")) {
      spaceTimer.current = setTimeout(() => setText((t) => (t && !t.endsWith(" ") ? t + " " : t)), SPACE_AFTER_MS);
    }
    return () => clearTimeout(spaceTimer.current);
  }, [paused, text]);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text.trim());
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="page">
      <h1>Free spelling</h1>
      <p className="hint">
        Sign anything; it's typed as you go. Lower your hand or pause for a moment to add a space.
        For double letters, slide your hand sideways.
      </p>
      <div className="free-text" aria-live="polite">
        {text || <span className="muted">Start signing...</span>}<span className="caret" />
      </div>
      <div className="toolbar">
        <button className="ghost" onClick={() => setText((t) => t.slice(0, -1))} disabled={!text}>⌫ Delete</button>
        <button className="ghost" onClick={() => setText("")} disabled={!text}>Clear</button>
        <button className="ghost" onClick={copy} disabled={!text.trim()}>{copied ? "Copied" : "Copy"}</button>
      </div>
      <div className="layout">
        <LiveCamera onFrame={onFrame} live={live} progress={progress} />
      </div>
    </div>
  );
}
