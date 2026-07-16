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
from .recognize import (
    binarize, extract_glyph, stroke_width, style_model_available, style_score,
)

INSET = 0.06  # fraction of each cell trimmed inside its boundaries before glyph search

# printed/handwritten discrimination thresholds
SAT_THRESH = 50         # min mean saturation of the high cluster before it counts as
                        # colored ink — an achromatic page never crosses this
SAT_GAP = 35            # min colored/achromatic saturation-cluster gap before trusting
                        # a split; tinted paper lifts every glyph together (one cluster)
PENCIL_GAP = 35         # min dark/light intensity-cluster gap before trusting a split
STYLE_GAP = 0.35        # min printed/handwritten style-score cluster gap before splitting
                        # (style_score has a per-scan offset, so the cut is adaptive)
SW_RATIO_GATE = 1.38    # min thick/thin stroke-width cluster ratio that signals two ink
                        # sources; fully-printed grids measure <= 1.31, mixed >= 1.45
STYLE_AGREE = 0.8       # the thin-stroke cluster must also style-score this much more
                        # handwritten on average, else the width split is warp/JPEG noise
                        # (clean grids <= +0.47, handwriting-bearing grids >= +1.44)

# Auto-select between the intensity/saturation filter and the style filter per grid.
# Intensity is more accurate on pencil / colored-ink pages and no-ops on clean grids;
# style only wins on dark-pen B&W scans where intensity can't separate the two ink
# sources. Defer to style only when it drops far more cells than intensity (a sign
# intensity stayed flat), else the style filter leaks handwriting on pencil pages.
STYLE_OVER_INTENSITY = 1.5   # style_drops must exceed this multiple of intensity_drops
STYLE_DROP_MARGIN = 10       # ...and exceed it by at least this many cells

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


def _achromatic_mask(saturations: list[float]) -> list[bool]:
    """True where a glyph's ink is achromatic (print-compatible).

    Saturation is judged per page, not against an absolute cut: tinted (aged) paper
    lifts every glyph's mean saturation together, so black print on a tan page can
    measure well above colored ink on a white one. An Otsu split drops the high
    cluster only when it is clearly separated (``SAT_GAP`` — one ink source forms a
    single tight cluster) *and* genuinely colored (``SAT_THRESH``).
    """
    if len(saturations) < 4:
        return [True] * len(saturations)
    vals = np.array(saturations, np.uint8)
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low, high = vals[vals <= thr], vals[vals > thr]
    if (low.size and high.size and (high.mean() - low.mean()) >= SAT_GAP
            and high.mean() >= SAT_THRESH):
        return (vals <= thr).tolist()
    return [True] * len(saturations)


def _printed_mask(intensities: list[float], saturations: list[float]) -> list[bool]:
    """Pick which glyphs are printed (dark, achromatic ink) vs handwritten.

    Colored ink (the separated high-saturation cluster) is dropped outright; among
    the achromatic remainder a 1-D Otsu split on intensity drops the light (pencil)
    cluster, but only when the two clusters are clearly separated — a fully-printed
    grid is one tight cluster and must be kept whole.
    """
    n = len(intensities)
    achromatic = _achromatic_mask(saturations)

    inten = [intensities[i] for i in range(n) if achromatic[i]]
    keep_dark = [True] * len(inten)
    if len(inten) >= 4:
        # Compare in the same quantized domain Otsu saw, or a printed glyph that
        # truncates onto the threshold (e.g. 50.6 vs thr 50) gets dropped.
        vals = np.array(inten, np.uint8)
        thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark, light = vals[vals <= thr], vals[vals > thr]
        if dark.size and light.size and (light.mean() - dark.mean()) >= PENCIL_GAP:
            keep_dark = (vals <= thr).tolist()

    out, k = [], 0
    for i in range(n):
        if not achromatic[i]:
            out.append(False)
        else:
            out.append(keep_dark[k])
            k += 1
    return out


def _kmeans2(values) -> np.ndarray:
    """1-D 2-means; boolean array marking the high cluster."""
    v = np.asarray(values, np.float64)
    c0, c1 = float(v.min()), float(v.max())
    if c0 == c1:
        return np.zeros(len(v), bool)
    for _ in range(50):
        hi = np.abs(v - c1) < np.abs(v - c0)
        if not hi.any() or hi.all():
            break
        nc0, nc1 = float(v[~hi].mean()), float(v[hi].mean())
        if (nc0, nc1) == (c0, c1):
            break
        c0, c1 = nc0, nc1
    return np.abs(v - c1) < np.abs(v - c0)


def _width_split(scores: list[float], widths: list[float]) -> np.ndarray | None:
    """Thick-stroke cluster mask, or ``None`` when the grid shows one ink source.

    Print and handwriting on one page come from different ink sources, so their
    stroke widths form two clusters. The split only fires when both gates pass:
    the thick/thin ratio clears ``SW_RATIO_GATE`` (a fully-printed grid is one tight
    cluster) *and* the thin cluster also style-scores more handwritten by
    ``STYLE_AGREE`` (warped or noisy print can vary in width, but then the two width
    clusters look equally printed to the style model).
    """
    w = np.asarray(widths, np.float64)
    sc = np.asarray(scores, np.float64)
    hi = _kmeans2(w)
    if (hi.any() and not hi.all()
            and w[hi].mean() >= SW_RATIO_GATE * w[~hi].mean()
            and sc[~hi].mean() - sc[hi].mean() >= STYLE_AGREE):
        return hi
    return None


def _style_keep(scores: list[float], widths: list[float]) -> list[bool]:
    """Keep the printed cluster using stroke width fused with the style score.

    The width-cluster gates (``_width_split``) are the sole trigger: without them
    the style score alone false-splits clean grids whose glyphs the model finds
    unfamiliar (e.g. themed digital fonts). Once fired, the thin-stroke cluster is
    dropped outright (it is the other ink source — a handwritten glyph whose style
    score happens to look printed must not be rescued), then glyphs are further
    split by 2-means on ``z(width) - z(score)`` (printed = thicker strokes + lower
    style score) — the z-scoring self-centers per grid, which the raw ``style_score``
    (per-scan offset) cannot do. The style score's largest-gap cut (``STYLE_GAP``)
    also applies on top — on a scan the model knows it removes the handwriting the
    width fusion lets through, and on an unseen scan with no clear gap it is a no-op.
    """
    if len(scores) < 4:
        return [True] * len(scores)
    keep = np.ones(len(scores), bool)
    hi = _width_split(scores, widths)
    if hi is not None:
        w = np.asarray(widths, np.float64)

        def z(v):
            v = np.asarray(v, np.float64)
            return (v - v.mean()) / (v.std() + 1e-9)
        keep &= hi
        keep &= _kmeans2(z(w) - z(scores))
        s = sorted(scores)
        gap, i = max((s[k + 1] - s[k], k) for k in range(len(s) - 1))
        if gap >= STYLE_GAP:
            keep &= np.asarray(scores) <= (s[i] + s[i + 1]) / 2
    return list(keep)


def _select_keep(glyphs: list, intensities: list[float], saturations: list[float],
                 stats: dict | None, force_intensity: bool) -> list[bool]:
    """Pick which present glyphs are printed givens, auto-selecting the filter.

    The style model's stroke-width gate is the handwriting *detector*: it stays silent
    on single-ink grids, so when it drops nothing the grid is clean (or all-printed) and
    every glyph is kept — the intensity split alone can't be trusted there, since clean
    print with tonal variation false-fires it. Once handwriting is confirmed, two
    regimes conflict and one is chosen: intensity + saturation for pencil / colored ink,
    or glyph shape (style) for B&W scans where dark handwriting matches print in both
    tone and hue. Style is trusted over intensity only when it drops far more cells (a
    sign intensity stayed flat), else it leaks handwriting on pencil pages.

    ``force_intensity`` skips the gate and pins the intensity/saturation path — the
    retry for when the auto-selected style filter blanks a whole grid.
    """
    intensity_keep = _printed_mask(intensities, saturations)
    if force_intensity or not style_model_available():
        if not force_intensity and stats is not None:
            stats["style_missing"] = True
        if stats is not None:
            stats["path"] = "intensity"
        return intensity_keep

    scores = [style_score(g) for g in glyphs]
    widths = [stroke_width(g) for g in glyphs]
    style_keep = _style_keep(scores, widths)
    sdrop = style_keep.count(False)
    if sdrop == 0:  # no handwriting detected -> keep all (ignore a lone intensity split)
        if stats is not None:
            stats["path"] = "none"
        return [True] * len(glyphs)
    idrop = intensity_keep.count(False)
    use_style = sdrop >= STYLE_OVER_INTENSITY * idrop and (sdrop - idrop) >= STYLE_DROP_MARGIN
    if stats is not None:
        stats["path"] = "style" if use_style else "intensity"
    if use_style:
        return style_keep
    # Handwriting is confirmed, so the thin-stroke cluster is the other ink source:
    # drop it even where intensity kept it (a firmly-drawn pencil stroke can measure
    # as dark as print, but its width can't match).
    hi = _width_split(scores, widths)
    if hi is not None:
        return [k and h for k, h in zip(intensity_keep, hi)]
    return intensity_keep


def iter_cells(warped: np.ndarray, color_warped: np.ndarray | None = None,
               stats: dict | None = None, force_intensity: bool = False):
    """Yield (index 0–80, glyph mask or None) for all 81 cells.

    A ``None`` glyph means the cell is empty *or* a handwritten answer: only printed
    givens survive. Light pencil is dropped via intensity, colored ink via saturation,
    and dark B&W handwriting via the style model — the filter is auto-selected per grid
    (see ``_select_keep``). ``color_warped`` is required for the saturation cue;
    ``force_intensity`` pins the intensity/saturation path (used as a retry when the
    auto-selected style filter blanks a whole grid).

    When ``stats`` is given it is filled with ``filtered`` (cells dropped as handwritten)
    and ``path`` (which filter was used) so callers can report what happened.
    """
    eq = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(warped)
    binary = binarize(eq)
    lines = grid_lines(binary)
    clean = cv2.subtract(binary, lines)
    # Features for printed/handwritten discrimination come from the *pre-CLAHE* warp,
    # since CLAHE destroys the absolute intensity that separates print from pencil.
    sat = (cv2.cvtColor(color_warped, cv2.COLOR_BGR2HSV)[:, :, 1]
           if color_warped is not None else None)

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
            if glyph is None:
                cells.append((idx, glyph, 0.0, 0.0))
                continue
            ink = glyph > 0
            mi = float(warped[y0:y1, x0:x1][ink].mean())
            ms = float(sat[y0:y1, x0:x1][ink].mean()) if sat is not None else 0.0
            cells.append((idx, glyph, mi, ms))

    present = [c for c in cells if c[1] is not None]
    keep = _select_keep([c[1] for c in present], [c[2] for c in present],
                        [c[3] for c in present], stats, force_intensity)
    drop = {present[i][0] for i in range(len(present)) if not keep[i]}
    if stats is not None:
        stats["filtered"] = len(drop)
    for idx, glyph, _, _ in cells:
        yield idx, (None if idx in drop else glyph)
