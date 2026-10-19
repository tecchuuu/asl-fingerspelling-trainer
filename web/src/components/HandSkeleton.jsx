import { HAND_CONNECTIONS } from "../lib/landmarker.js";
import SKELETONS from "../lib/skeletons.json";

/**
 * A letter drawn as its hand skeleton: the typical shape from the training data
 * (scripts/make_skeletons.py), the same 21 points and bones the camera overlay draws.
 */
export default function HandSkeleton({ letter, size = 120, title }) {
  const pts = SKELETONS[letter];
  if (!pts) return null;
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const pad = 0.25;
  const [minX, minY] = [Math.min(...xs) - pad, Math.min(...ys) - pad];
  const w = Math.max(...xs) - minX + pad;
  const h = Math.max(...ys) - minY + pad;
  const side = Math.max(w, h);
  // centre the hand in a square box
  const ox = minX - (side - w) / 2;
  const oy = minY - (side - h) / 2;
  return (
    <svg className="skeleton" width={size} height={size} viewBox={`${ox} ${oy} ${side} ${side}`}
      role={title ? "img" : undefined} aria-label={title} aria-hidden={title ? undefined : true}>
      {HAND_CONNECTIONS.map(({ start, end }, i) => (
        <line key={i} x1={pts[start][0]} y1={pts[start][1]} x2={pts[end][0]} y2={pts[end][1]} className="bone" />
      ))}
      {pts.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={i === 0 ? 0.08 : 0.055} className="joint" />
      ))}
    </svg>
  );
}
