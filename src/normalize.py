"""Turn MediaPipe's raw landmarks into hand-shape features.

Used by training AND the live app, so both see exactly the same numbers.
"""

import numpy as np

WRIST = 0
MIDDLE_MCP = 9  # knuckle at the base of the middle finger


def normalize(landmarks, aspect=1.0):
    """(..., 21, 3) raw landmarks -> (..., 63) features.

    aspect: image width / height. MediaPipe divides x by the width and y by the height,
    so on a non-square webcam frame the hand looks stretched unless x is corrected.
    """
    lm = np.asarray(landmarks, dtype=np.float32).copy()
    # z is on the same scale as x
    lm[..., 0] *= aspect
    lm[..., 2] *= aspect

    lm -= lm[..., WRIST:WRIST + 1, :]

    # Palm length doesn't change with finger pose, unlike e.g. the hand's bounding box
    palm = np.linalg.norm(lm[..., MIDDLE_MCP, :], axis=-1, keepdims=True)
    lm /= palm[..., None]

    return lm.reshape(*lm.shape[:-2], -1)


def mirror(features):
    """Flip normalized features left <-> right, turning a right hand into a left hand."""
    lm = np.asarray(features, dtype=np.float32).reshape(*np.shape(features)[:-1], 21, 3).copy()
    lm[..., 0] *= -1
    return lm.reshape(np.shape(features))
