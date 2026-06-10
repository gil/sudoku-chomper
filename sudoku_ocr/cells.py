"""Split a warped grid into 81 cells and isolate each cell's digit glyph.

The whole grid is binarized once with a global Otsu threshold so faint
bleed-through / paper texture stays background; per-cell processing then only
keeps the largest interior component.

Cell boundaries come from the *actual* detected grid lines when all 10 are found
(robust to perspective leaving non-uniform cell spacing), falling back to an even
9-way split of the warp.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detect import SIZE
from .recognize import binarize, extract_glyph

INSET = 0.06  # fraction of each cell trimmed inside its boundaries before glyph search

# Grid lines run continuously across the whole warped grid, so a long 1-D opening
# isolates them while leaving digits (which never span this length) intact. Removing
# them stops skewed line fragments from being read as "1"s.
_LINE_LEN = max(15, SIZE // 10)


def grid_lines(binary: np.ndarray) -> np.ndarray:
    """Mask of the long horizontal + vertical lines in a binarized grid."""
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (_LINE_LEN, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, _LINE_LEN))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    lines |= cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)
    return cv2.dilate(lines, np.ones((3, 3), np.uint8))


def _line_centers(profile: np.ndarray) -> list[float]:
    """Sub-pixel centers of the line peaks in a 1-D projection."""
    thr = profile.max() * 0.25
    if thr <= 0:
        return []
    on = profile > thr
    centers: list[float] = []
    i, n = 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            seg = profile[i:j].astype(np.float64)
            centers.append(i + float((np.arange(j - i) * seg).sum() / seg.sum()))
            i = j
        else:
            i += 1
    return centers


def _ten_boundaries(profile: np.ndarray) -> list[float]:
    """10 cell boundaries from a line projection, degrading gracefully:

    10 lines -> use directly; 4 -> box separators at 0/3/6/9, interpolate within each
    band; >=2 -> uniform across the true detected extent (trims outer-frame offset);
    else -> even split of the whole warp.
    """
    even = [i * SIZE / 9.0 for i in range(10)]
    c = _line_centers(profile)
    # Only trust detected lines when they actually span the warp (first line near the
    # left/top edge, last near the right/bottom) — a couple of faint interior lines
    # must not be mistaken for the grid extent.
    if not c or c[0] > 0.12 * SIZE or c[-1] < 0.88 * SIZE:
        return even
    if len(c) == 10:
        return c
    if len(c) == 4:  # box separators at 0/3/6/9 -> interpolate the 2 lines per band
        out: list[float] = []
        for k in range(3):
            a, b = c[k], c[k + 1]
            out += [a + (b - a) * t / 3 for t in range(3)]
        return out + [c[3]]
    return even


def iter_cells(warped: np.ndarray):
    """Yield (index 0–80, glyph mask or None) for all 81 cells.

    A ``None`` glyph means the cell is empty.
    """
    eq = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(warped)
    binary = binarize(eq)
    lines = grid_lines(binary)
    clean = cv2.subtract(binary, lines)

    bx = _ten_boundaries(lines.sum(axis=0))
    by = _ten_boundaries(lines.sum(axis=1))

    for row in range(9):
        for col in range(9):
            ch = by[row + 1] - by[row]
            cw = bx[col + 1] - bx[col]
            y0 = int(by[row] + ch * INSET)
            y1 = int(by[row + 1] - ch * INSET)
            x0 = int(bx[col] + cw * INSET)
            x1 = int(bx[col + 1] - cw * INSET)
            glyph = extract_glyph(clean[y0:y1, x0:x1])
            yield row * 9 + col, glyph
