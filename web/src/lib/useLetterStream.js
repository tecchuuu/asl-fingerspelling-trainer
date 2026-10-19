import { useCallback, useRef, useState } from "react";
import { predict } from "./api.js";
import { Gate, handGeometry } from "./gate.js";

/**
 * Camera frames -> /api/predict -> gate. Shared by every practice mode.
 * Returns { onFrame, live, progress, resetGate }:
 *   onFrame:   pass to <Camera onFrame={...}>
 *   live:      latest prediction { letter, confidence, top3 } or null (no hand)
 *   progress:  0..1, how close the current letter is to being typed
 * onTyped(letter, prediction) is called whenever the gate types a letter.
 */
export function useLetterStream(onTyped) {
  const gate = useRef(new Gate());
  const inFlight = useRef(false);
  const onTypedRef = useRef(onTyped);
  const [live, setLive] = useState(null);
  const [progress, setProgress] = useState(0);
  onTypedRef.current = onTyped;

  const handle = useCallback((prediction, geometry) => {
    const typed = gate.current.update(prediction, geometry?.wristX, geometry?.palm);
    setLive(prediction);
    setProgress(gate.current.progress);
    if (typed) onTypedRef.current?.(typed, prediction);
  }, []);

  const onFrame = useCallback(({ landmarks, width, height }) => {
    if (!landmarks) return handle(null, null);
    if (inFlight.current) return; // one request at a time; frames in between are skipped
    inFlight.current = true;
    const geometry = handGeometry(landmarks, width, height);
    predict(landmarks, width, height)
      .then((p) => handle(p, geometry))
      .catch(() => handle(null, null))
      .finally(() => (inFlight.current = false));
  }, [handle]);

  const resetGate = useCallback(() => {
    gate.current.reset();
    setProgress(0);
  }, []);

  return { onFrame, live, progress, resetGate };
}
