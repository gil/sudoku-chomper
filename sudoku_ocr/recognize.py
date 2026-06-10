"""Printed-digit recognition: glyph normalization, HOG features, SVM inference.

The normalization + HOG pipeline here is shared with ``train.py`` so that training
samples and live cell crops land in the same feature distribution.
"""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
from skimage.feature import hog

GLYPH = 28           # normalized glyph canvas (px)
INNER = 20           # digit is fit into this box, then centered on the canvas

MODEL_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "models", "digit_svm.joblib")


def binarize(gray: np.ndarray) -> np.ndarray:
    """Return a uint8 mask where ink == 255, background == 0 (Otsu, auto-polarity)."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # THRESH_BINARY_INV assumes dark ink on light bg. If the result is mostly white,
    # the source was light-on-dark, so flip back.
    if mask.mean() > 127:
        mask = cv2.bitwise_not(mask)
    return mask


def extract_glyph(bin_cell: np.ndarray) -> np.ndarray | None:
    """Isolate the digit from a *pre-binarized* cell; ``None`` if the cell is empty.

    The grid-line frame is erased and only the largest interior connected component
    is kept, so paper texture, faint bleed-through, noise speckles, and border-line
    fragments don't register as a digit. Drives both empty-detection and recognition.

    The caller binarizes the whole grid once (global Otsu), which keeps faint
    bleed-through ghosts as background — per-cell Otsu would wrongly promote them.
    """
    mask = bin_cell.copy()
    h, w = mask.shape
    if h < 6 or w < 6:
        return None
    f = max(1, int(0.10 * min(h, w)))  # erase the grid-line frame
    mask[:f] = 0
    mask[-f:] = 0
    mask[:, :f] = 0
    mask[:, -f:] = 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < 0.012 * h * w:  # too small -> empty cell
        return None
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def normalize_glyph(mask: np.ndarray) -> np.ndarray | None:
    """Crop the glyph from an ink mask and center it on a GLYPH×GLYPH canvas.

    Returns ``None`` when the mask holds no meaningful ink (treated as empty).
    """
    ys, xs = np.where(mask > 0)
    if xs.size < 8:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    glyph = mask[y0:y1 + 1, x0:x1 + 1]
    h, w = glyph.shape
    scale = INNER / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    glyph = cv2.resize(glyph, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((GLYPH, GLYPH), np.uint8)
    oy, ox = (GLYPH - nh) // 2, (GLYPH - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = glyph
    return canvas


def hog_features(canvas: np.ndarray) -> np.ndarray:
    """Feature vector: HOG (stroke orientation) + downsampled pixels (loop fill).

    HOG alone misses whether the interior of 6/8/9/0 is open or filled; appending a
    coarse pixel grid gives the classifier that holistic shape cue, which sharply
    cuts 1/3/6/8/9 confusion on bold low-res print.
    """
    norm = canvas.astype(np.float32) / 255.0
    h = hog(
        norm,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    pix = cv2.resize(norm, (14, 14), interpolation=cv2.INTER_AREA).ravel()
    return np.concatenate([h, pix])


@lru_cache(maxsize=1)
def _load_model():
    import joblib

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Digit model not found at {MODEL_PATH}. Run: python -m sudoku_ocr.train"
        )
    return joblib.load(MODEL_PATH)


def predict_glyph(glyph: np.ndarray) -> int:
    """Classify an isolated glyph mask (from ``extract_glyph``) as 1–9 (0 if empty)."""
    canvas = normalize_glyph(glyph)
    if canvas is None:
        return 0
    return int(_load_model().predict([hog_features(canvas)])[0])
