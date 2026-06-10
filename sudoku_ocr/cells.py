"""Split a warped grid into 81 cells and isolate each cell's digit glyph.

The whole grid is binarized once with a global Otsu threshold so faint
bleed-through / paper texture stays background; per-cell processing then only
keeps the largest interior component.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detect import SIZE
from .recognize import binarize, extract_glyph

STEP = SIZE / 9.0
MARGIN = 0.06  # slack kept around each cell; extract_glyph frames out the grid lines

# Grid lines run continuously across the whole warped grid, so a long 1-D opening
# isolates them while leaving digits (which never span this length) intact. Removing
# them stops skewed line fragments from being read as "1"s.
_LINE_LEN = max(15, SIZE // 10)


def remove_grid_lines(binary: np.ndarray) -> np.ndarray:
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (_LINE_LEN, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, _LINE_LEN))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    lines |= cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))  # also clear anti-aliased edges
    return cv2.subtract(binary, lines)


def _cell_box(row: int, col: int) -> tuple[int, int, int, int]:
    y0 = int(row * STEP + STEP * MARGIN)
    y1 = int((row + 1) * STEP - STEP * MARGIN)
    x0 = int(col * STEP + STEP * MARGIN)
    x1 = int((col + 1) * STEP - STEP * MARGIN)
    return y0, y1, x0, x1


def iter_cells(warped: np.ndarray):
    """Yield (index 0–80, glyph mask or None) for all 81 cells.

    A ``None`` glyph means the cell is empty.
    """
    eq = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(warped)
    binary = remove_grid_lines(binarize(eq))
    for row in range(9):
        for col in range(9):
            y0, y1, x0, x1 = _cell_box(row, col)
            glyph = extract_glyph(binary[y0:y1, x0:x1])
            yield row * 9 + col, glyph
