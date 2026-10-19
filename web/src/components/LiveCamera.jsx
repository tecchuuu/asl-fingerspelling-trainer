import Camera from "./Camera.jsx";

/** Camera plus the model's live guess and the gate's progress bar. */
export default function LiveCamera({ onFrame, live, progress }) {
  return (
    <div>
      <Camera onFrame={onFrame} />
      <div className="guess">
        <span>{live ? (live.letter === "other" ? "–" : live.letter) : "no hand"}</span>
        <div className="bar"><div style={{ width: `${progress * 100}%` }} /></div>
      </div>
    </div>
  );
}
