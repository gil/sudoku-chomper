"""Locate Sudoku grids in an image and perspective-correct each to a square.

Supports multiple grids per image (e.g. two stacked puzzles, or a puzzle plus its
printed solution). Returns warped grayscale crops ordered top-to-bottom,
left-to-right.
"""

from __future__ import annotations

import cv2
import numpy as np

SIZE = 900  # side length of each warped grid (px); high enough to keep glyph detail


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _warp(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst = np.array([[0, 0], [SIZE - 1, 0], [SIZE - 1, SIZE - 1], [0, SIZE - 1]], np.float32)
    M = cv2.getPerspectiveTransform(_order_corners(corners), dst)
    return cv2.warpPerspective(gray, M, (SIZE, SIZE))


def _line_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask of dark structure (grid lines / ink), polarity-agnostic."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    # Connect faint / broken grid lines (newspaper scans).
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def _square_candidates(mask: np.ndarray, img_area: float) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out: list[np.ndarray] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.03 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        if min(w, h) == 0 or not 0.7 <= w / h <= 1.4:
            continue
        out.append(approx)
    return out


def _dedup(cands: list[np.ndarray]) -> list[np.ndarray]:
    """Drop nested duplicates (inner vs outer border) by center proximity."""
    kept: list[tuple[np.ndarray, tuple[float, float], float]] = []
    for c in sorted(cands, key=cv2.contourArea, reverse=True):
        (cx, cy), (w, h), _ = cv2.minAreaRect(c)
        r = max(w, h)
        if any(abs(cx - kx) < kr * 0.5 and abs(cy - ky) < kr * 0.5 for _, (kx, ky), kr in kept):
            continue
        kept.append((c, (cx, cy), r))
    return [c for c, _, _ in kept]


def _content_bbox(mask: np.ndarray) -> np.ndarray | None:
    """Fallback: bounding box of all dark content (tight digital crops)."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if (x1 - x0) < mask.shape[1] * 0.4 or (y1 - y0) < mask.shape[0] * 0.4:
        return None
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def find_grids(image: np.ndarray) -> list[np.ndarray]:
    """Return warped grayscale grid crops (SIZE×SIZE), ordered for reading."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = _line_mask(gray)
    cands = _dedup(_square_candidates(mask, gray.shape[0] * gray.shape[1]))

    if not cands:
        box = _content_bbox(mask)
        cands = [box] if box is not None else []

    grids = []
    for c in cands:
        cx, cy = c.reshape(-1, 2).mean(axis=0)
        grids.append((cy, cx, _warp(gray, c)))

    # Top-to-bottom, left-to-right: bucket centers into row bands, then by x.
    band = max(1.0, image.shape[0] * 0.15)
    grids.sort(key=lambda g: (g[0] // band, g[1]))
    return [g[2] for g in grids]


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img
