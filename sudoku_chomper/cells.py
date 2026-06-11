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
from .recognize import binarize, extract_glyph, style_model_available, style_score

INSET = 0.06  # fraction of each cell trimmed inside its boundaries before glyph search

# printed/handwritten discrimination (only used when printed_only is requested)
SAT_THRESH = 50         # mean HSV saturation above this = colored ink (pen), not print
PENCIL_GAP = 35         # min dark/light intensity-cluster gap before trusting a split
STYLE_GAP = 0.35        # min printed/handwritten style-score cluster gap before splitting
                        # (style_score has a per-scan offset, so the cut is adaptive)

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


def _printed_mask(intensities: list[float], saturations: list[float]) -> list[bool]:
    """Pick which glyphs are printed (dark, achromatic ink) vs handwritten.

    Colored ink (high saturation) is dropped outright; among the achromatic remainder
    a 1-D Otsu split on intensity drops the light (pencil) cluster, but only when the
    two clusters are clearly separated — a fully-printed grid is one tight cluster and
    must be kept whole.
    """
    n = len(intensities)
    achromatic = [s < SAT_THRESH for s in saturations]

    inten = [intensities[i] for i in range(n) if achromatic[i]]
    keep_dark = [True] * len(inten)
    if len(inten) >= 4:
        vals = np.array(inten, np.uint8).reshape(-1, 1)
        thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = [v for v in inten if v <= thr]
        light = [v for v in inten if v > thr]
        if dark and light and (np.mean(light) - np.mean(dark)) >= PENCIL_GAP:
            keep_dark = [v <= thr for v in inten]

    out, k = [], 0
    for i in range(n):
        if not achromatic[i]:
            out.append(False)
        else:
            out.append(keep_dark[k])
            k += 1
    return out


def _style_keep(scores: list[float]) -> list[bool]:
    """Keep the lower (more printed) style-score cluster, adaptively.

    ``style_score`` carries a per-scan offset, so a fixed cutoff is unreliable; a 1-D
    split self-centers per grid. The printed and handwritten scores form two clusters
    separated by a gap, so cut at the largest gap and keep the lower side. Only splits
    when that gap is clear, so a single-style grid (all printed) is kept whole.
    """
    if len(scores) < 4:
        return [True] * len(scores)
    s = sorted(scores)
    gap, i = max((s[k + 1] - s[k], k) for k in range(len(s) - 1))
    if gap < STYLE_GAP:
        return [True] * len(scores)
    cut = (s[i] + s[i + 1]) / 2
    return [v <= cut for v in scores]


def iter_cells(warped: np.ndarray, printed_only: bool = False,
               color_warped: np.ndarray | None = None, use_style: bool = False,
               stats: dict | None = None):
    """Yield (index 0–80, glyph mask or None) for all 81 cells.

    A ``None`` glyph means the cell is empty. With ``printed_only`` (needs
    ``color_warped``), handwritten answers are returned as ``None`` so only printed
    givens survive: light pencil via intensity, colored ink via saturation, and — for
    black-and-white scans where neither fires — handwritten *shape* via the style model
    (``use_style``).

    When ``stats`` is given it is filled with ``filtered`` (cells dropped as handwritten)
    so callers can report that filtering happened.
    """
    eq = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(warped)
    binary = binarize(eq)
    lines = grid_lines(binary)
    clean = cv2.subtract(binary, lines)
    # Features for printed/handwritten discrimination come from the *pre-CLAHE* warp,
    # since CLAHE destroys the absolute intensity that separates print from pencil.
    sat = (cv2.cvtColor(color_warped, cv2.COLOR_BGR2HSV)[:, :, 1]
           if printed_only and color_warped is not None else None)

    bx = _ten_boundaries(lines.sum(axis=0))
    by = _ten_boundaries(lines.sum(axis=1))

    cells = []  # (index, glyph, mean_intensity, mean_saturation)
    for row in range(9):
        for col in range(9):
            ch = by[row + 1] - by[row]
            cw = bx[col + 1] - bx[col]
            y0 = int(by[row] + ch * INSET)
            y1 = int(by[row + 1] - ch * INSET)
            x0 = int(bx[col] + cw * INSET)
            x1 = int(bx[col + 1] - cw * INSET)
            glyph = extract_glyph(clean[y0:y1, x0:x1])
            idx = row * 9 + col
            if not printed_only or glyph is None:
                cells.append((idx, glyph, 0.0, 0.0))
                continue
            ink = glyph > 0
            mi = float(warped[y0:y1, x0:x1][ink].mean())
            ms = float(sat[y0:y1, x0:x1][ink].mean()) if sat is not None else 0.0
            cells.append((idx, glyph, mi, ms))

    if printed_only:
        present = [c for c in cells if c[1] is not None]
        # Two regimes: shape (style) for B&W scans where dark handwriting matches print
        # in intensity *and* hue (and JPEG chroma fringing makes saturation unreliable);
        # intensity+saturation for pencil / colored ink. They conflict, so pick one.
        if use_style and style_model_available():
            keep = _style_keep([style_score(c[1]) for c in present])
        else:
            if use_style and stats is not None:
                stats["style_missing"] = True  # fall back to intensity/saturation
            keep = _printed_mask([c[2] for c in present], [c[3] for c in present])
        drop = {present[i][0] for i in range(len(present)) if not keep[i]}
        if stats is not None:
            stats["filtered"] = len(drop)
        for idx, glyph, _, _ in cells:
            yield idx, (None if idx in drop else glyph)
    else:
        for idx, glyph, _, _ in cells:
            yield idx, glyph
