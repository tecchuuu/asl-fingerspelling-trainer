"""Landmarks -> letter probabilities, the way the live demo and the app call the MLP."""

import numpy as np
import torch

from src import config
from src.dataset import pinch
from src.normalize import normalize

# The "other" class was trained only on my webcam (camera above, looking down), so it also
# swallowed my F's (three spread fingers look like an open hand from that angle). A hand whose
# thumb tip touches the index tip (F, O, D, ...) is making a shape, not resting, so it may not
# be "other". 95% of dataset F's are within 0.31 palm lengths; non-letter hands are mostly > 0.4.
PINCH_GUARD = 0.35


@torch.no_grad()
def letter_probs(model, raw_landmarks, aspect=1.0):
    """(..., 21, 3) raw MediaPipe landmarks -> (..., n_classes) probabilities.
    For the 25-class model, "other" is ruled out when the thumb and index tips touch."""
    features = normalize(raw_landmarks, aspect=aspect)
    single = features.ndim == 1
    features = np.atleast_2d(features)
    probs = torch.softmax(model(torch.from_numpy(features).float()), dim=1).numpy()
    if probs.shape[1] > config.NUM_CLASSES:
        pinched = pinch(features) < PINCH_GUARD
        probs[pinched, config.OTHER_INDEX] = 0
        probs[pinched] /= probs[pinched].sum(axis=1, keepdims=True)
    return probs[0] if single else probs
