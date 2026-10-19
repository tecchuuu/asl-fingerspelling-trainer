import { useRef, useState } from "react";
import Camera from "../components/Camera.jsx";
import { compareWithPython, predict } from "../lib/api.js";

/**
 * Phase 4 step 1: does MediaPipe in the browser give the same hand (and letter) as in Python?
 * Live: browser landmarks -> /api/predict. On "Compare": the same frame is also sent as an image,
 * Python runs its own MediaPipe on it, and both predictions are shown side by side.
 */
export default function CheckPage() {
  const latest = useRef(null);
  const inFlight = useRef(false);
  const [live, setLive] = useState(null);
  const [rows, setRows] = useState([]);
  const [busy, setBusy] = useState(false);

  function onFrame(frame) {
    latest.current = frame;
    if (!frame.landmarks) return setLive(null);
    if (inFlight.current) return; // one request at a time; skip frames meanwhile
    inFlight.current = true;
    predict(frame.landmarks, frame.width, frame.height)
      .then(setLive)
      .catch((err) => setLive({ error: err.message }))
      .finally(() => (inFlight.current = false));
  }

  async function compare() {
    const frame = latest.current;
    if (!frame?.landmarks) return;
    setBusy(true);
    const canvas = document.createElement("canvas");
    canvas.width = frame.width;
    canvas.height = frame.height;
    canvas.getContext("2d").drawImage(frame.video, 0, 0); // raw frame, not mirrored
    try {
      const r = await compareWithPython(frame.landmarks, frame.width, frame.height,
        canvas.toDataURL("image/jpeg", 0.95));
      setRows((prev) => [...prev, r]);
    } catch (err) {
      setRows((prev) => [...prev, { error: err.message }]);
    } finally {
      setBusy(false);
    }
  }

  const done = rows.filter((r) => r.python_found_hand);
  const same = done.filter((r) => r.browser.letter === r.python.letter).length;

  return (
    <div className="page">
      <h1>Browser vs Python check</h1>
      <p className="hint">
        Sign a letter, hold it, press <b>Compare</b>. Do this for about 5 different letters.
        Browser and Python should agree on the letter, with landmarks only a few pixels apart.
      </p>
      <div className="layout">
        <Camera onFrame={onFrame} />
        <div className="panel">
          <div className="live">
            {live?.error ? <span className="bad">{live.error}</span>
              : live ? <><span className="big">{live.letter}</span> {(live.confidence * 100).toFixed(0)}%</>
              : <span className="muted">no hand</span>}
          </div>
          <button onClick={compare} disabled={busy || !live || live.error}>
            {busy ? "Comparing..." : "Compare with Python"}
          </button>
          {done.length > 0 && (
            <p><b>{same}/{done.length}</b> same letter</p>
          )}
          <table>
            <thead>
              <tr><th>#</th><th>Browser</th><th>Python</th><th>xy diff</th></tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  {r.error ? <td colSpan={3} className="bad">{r.error}</td>
                    : !r.python_found_hand ? <><td>{r.browser.letter}</td><td colSpan={2} className="bad">Python found no hand</td></>
                    : <>
                        <td>{r.browser.letter} {(r.browser.confidence * 100).toFixed(0)}%</td>
                        <td className={r.browser.letter === r.python.letter ? "" : "bad"}>
                          {r.python.letter} {(r.python.confidence * 100).toFixed(0)}%
                        </td>
                        <td>{r.mean_xy_diff_px.toFixed(1)} px</td>
                      </>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
