// Confidence gate: turns per-frame predictions into typed letters.
// Same rules and settings as src/gate.py (checked by scripts/check-gate.mjs):
//   1. A letter is typed once it wins `stableFrames` frames in a row with confidence >= `minConf`.
//   2. Holding it doesn't type it again. The gate re-arms when another letter is typed
//      or the hand is gone for `releaseFrames` frames.
//   3. Double letters: holding the same letter and sliding the wrist sideways by
//      `doubleShift` palm lengths types it again.
// "other" (a hand that isn't signing) counts like an unsure frame: no streak, no typing.

export const OTHER = "other";

export const DEFAULT_SETTINGS = {
  stableFrames: 10,
  minConf: 0.8,
  releaseFrames: 5,
  doubleShift: 0.6,
};

export class Gate {
  constructor(settings = DEFAULT_SETTINGS) {
    this.s = settings;
    this.reset();
  }

  reset() {
    this.candidate = null;
    this.streak = 0;
    this.typed = null;
    this.anchorX = null;
    this.missing = 0;
  }

  /** 0..1, how close the current candidate is to being typed (for the progress bar). */
  get progress() {
    return Math.min(this.streak / this.s.stableFrames, 1);
  }

  /**
   * One frame. prediction: { letter, confidence } or null when no hand is found.
   * wristX, palm: wrist x and palm length in pixels, for the double-letter rule.
   * Returns the typed letter, or null.
   */
  update(prediction, wristX = null, palm = null) {
    if (!prediction) {
      this.candidate = null;
      this.streak = 0;
      this.missing += 1;
      if (this.missing >= this.s.releaseFrames) this.typed = null;
      return null;
    }
    this.missing = 0;

    const { letter, confidence } = prediction;
    if (letter === OTHER || confidence < this.s.minConf) {
      this.candidate = null;
      this.streak = 0;
      return null;
    }

    this.streak = letter === this.candidate ? this.streak + 1 : 1;
    this.candidate = letter;
    if (this.streak < this.s.stableFrames) return null;

    if (letter !== this.typed) return this.#type(letter, wristX);
    if (wristX != null && palm && this.anchorX != null
        && Math.abs(wristX - this.anchorX) / palm >= this.s.doubleShift) {
      return this.#type(letter, wristX);
    }
    return null;
  }

  #type(letter, wristX) {
    this.typed = letter;
    this.anchorX = wristX;
    return letter;
  }
}

/** Wrist x and palm length (wrist -> middle-finger knuckle) in pixels, for the double-letter rule. */
export function handGeometry(landmarks, width, height) {
  const wrist = landmarks[0];
  const mcp = landmarks[9];
  return {
    wristX: wrist.x * width,
    palm: Math.hypot((mcp.x - wrist.x) * width, (mcp.y - wrist.y) * height),
  };
}
