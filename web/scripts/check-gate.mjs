// Same cases as scripts/check_gate.py, so the browser gate behaves exactly like the Python one.
// Run from web/:  node scripts/check-gate.mjs
import { Gate, OTHER } from "../src/lib/gate.js";

const PALM = 100;
const frame = (letter, confidence = 0.95, wristX = 300) => [{ letter, confidence }, wristX];
const NO_HAND = [null, null];
const rep = (f, n) => Array.from({ length: n }, () => f);

const CASES = {
  "hold A for 2 s types one A": [rep(frame("A"), 60), "A"],
  "A then B": [[...rep(frame("A"), 15), ...rep(frame("B"), 15)], "AB"],
  "flicker between letters is ignored": [[...rep(frame("A"), 15), frame("S"), frame("E"), frame("S"), ...rep(frame("B"), 15)], "AB"],
  "9 stable frames is not enough": [rep(frame("A"), 9), ""],
  "low confidence never types": [rep(frame("A", 0.6), 30), ""],
  "hand leaves, same letter again": [[...rep(frame("A"), 15), ...rep(NO_HAND, 6), ...rep(frame("A"), 15)], "AA"],
  "brief hand loss doesn't re-type": [[...rep(frame("A"), 15), ...rep(NO_HAND, 3), ...rep(frame("A"), 15)], "A"],
  "low-confidence blip doesn't re-type": [[...rep(frame("A"), 15), ...rep(frame("A", 0.5), 2), ...rep(frame("A"), 15)], "A"],
  "slide 0.8 palm sideways = double": [[...rep(frame("L"), 15), ...rep(frame("L", 0.95, 380), 15)], "LL"],
  "small drift 0.3 palm is not a double": [[...rep(frame("L"), 15), ...rep(frame("L", 0.95, 330), 15)], "L"],
  "an open hand ('other') never types": [rep(frame(OTHER), 60), ""],
  "other between two holds doesn't re-type": [[...rep(frame("A"), 15), ...rep(frame(OTHER), 20), ...rep(frame("A"), 15)], "A"],
  "other then a new letter": [[...rep(frame(OTHER), 20), ...rep(frame("B"), 15)], "B"],
  "HELLO": [[...rep(frame("H"), 12), ...rep(frame("E"), 12), ...rep(frame("L"), 12),
             ...rep(frame("L", 0.95, 370), 12), ...rep(frame("O"), 12)], "HELLO"],
};

let failed = 0;
for (const [name, [frames, want]] of Object.entries(CASES)) {
  const gate = new Gate();
  const got = frames.map(([p, x]) => gate.update(p, x, PALM) ?? "").join("");
  if (got !== want) failed += 1;
  console.log(`${got === want ? "PASS" : "FAIL"}  ${name.padEnd(40)} expected '${want}', got '${got}'`);
}
console.log(failed ? `${failed} failed.` : "All passed.");
process.exit(failed ? 1 : 0);
