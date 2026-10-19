"""Confidence gate: turns per-frame predictions into typed letters.

Rules:
  1. A letter is typed once it wins STABLE_FRAMES frames in a row with confidence >= MIN_CONF.
  2. Holding it doesn't type it again. The gate re-arms when another letter is typed
     or the hand is gone for RELEASE_FRAMES frames.
  3. Double letters (LL, SS): holding the same letter and sliding the hand sideways by
     DOUBLE_SHIFT palm lengths types it again, the way signers do double letters.

Used by the live demo and the web app, so both behave the same.
"""

from dataclasses import dataclass

import numpy as np

from src import config


@dataclass
class GateSettings:
    stable_frames: int = 10
    min_conf: float = 0.8
    release_frames: int = 5
    double_shift: float = 0.6  # in palm lengths, so it works at any distance from the camera


class Gate:
    def __init__(self, settings=None):
        self.s = settings or GateSettings()
        self.candidate = None   # letter index currently building a streak
        self.streak = 0
        self.typed = None       # last typed letter index, blocks re-typing while held
        self.anchor_x = None    # wrist x when it was typed, for the double-letter slide
        self.missing = 0

    @property
    def progress(self):
        """0..1 how close the current candidate is to being typed (for a progress ring in the UI)."""
        return min(self.streak / self.s.stable_frames, 1.0)

    def update(self, probs, wrist_x=None, palm=None):
        """One frame. probs: (24,) or (25,) softmax output (25th = "other"), or None when no hand is found.
        wrist_x, palm: wrist x position and palm length in pixels, for the double-letter rule.
        Returns the typed letter, or None."""
        if probs is None:
            self.candidate, self.streak = None, 0
            self.missing += 1
            if self.missing >= self.s.release_frames:
                self.typed = None
            return None
        self.missing = 0

        top = int(np.argmax(probs))
        # "other" (a hand that isn't signing) counts like an unsure frame: no streak, no typing.
        # It doesn't re-arm the gate either; only a hand leaving the frame or a new letter does.
        if top == config.OTHER_INDEX or probs[top] < self.s.min_conf:
            self.candidate, self.streak = None, 0
            return None

        self.streak = self.streak + 1 if top == self.candidate else 1
        self.candidate = top
        if self.streak < self.s.stable_frames:
            return None

        if top != self.typed:
            return self._type(top, wrist_x)
        if wrist_x is not None and palm and self.anchor_x is not None:
            if abs(wrist_x - self.anchor_x) / palm >= self.s.double_shift:
                return self._type(top, wrist_x)
        return None

    def _type(self, letter, wrist_x):
        self.typed = letter
        self.anchor_x = wrist_x
        return config.LETTERS[letter]
