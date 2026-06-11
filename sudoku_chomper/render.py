"""Render an 81-char Sudoku string back to a clean digital grid image."""

from __future__ import annotations

import cv2
import numpy as np

CELL = 64
MARGIN = 16
SIZE = CELL * 9 + MARGIN * 2


def render_puzzle(puzzle: str) -> np.ndarray:
    img = np.full((SIZE, SIZE), 255, dtype=np.uint8)

    for k in range(10):
        thick = 3 if k % 3 == 0 else 1
        p = MARGIN + k * CELL
        cv2.line(img, (MARGIN, p), (SIZE - MARGIN, p), 0, thick)
        cv2.line(img, (p, MARGIN), (p, SIZE - MARGIN), 0, thick)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4
    weight = 2
    for i, ch in enumerate(puzzle):
        if ch == "0":
            continue
        r, c = divmod(i, 9)
        (tw, th), _ = cv2.getTextSize(ch, font, scale, weight)
        x = MARGIN + c * CELL + (CELL - tw) // 2
        y = MARGIN + r * CELL + (CELL + th) // 2
        cv2.putText(img, ch, (x, y), font, scale, 0, weight, cv2.LINE_AA)

    return img
