# Improvements log

Record of what was tried to improve Sudoku grid detection + digit OCR, why, the
result, and what is still worth trying. Parked here for future work.

## Current accuracy (bundled samples)

| Sample | Type | Result |
|---|---|---|
| `sample.png` | digital screenshot | exact |
| `Screenshot …20.11.58` / `…20.12.10` | printed book scans | exact |
| `Screenshot …20.13.58` | two stacked puzzles | both exact |
| `0-sudoku.jpg` | photo of web printout | exact |
| `Sudoku.jpg` | ~30° rotated torn clipping | exact |
| `30be7414…jpg` | 4 puzzles/page, colored bg | all 4 exact |
| `sudoku-evil-5.jpg` | 4 puzzles/page + outer frame | all 4 exact (frame filtered) |
| `s-l1200.png` | 2 puzzles/page | exact |
| `sudoku_solved.png` | puzzle + solution side-by-side | puzzle exact (solution suppressed) |
| `sudoku-warped-example.png` | warped + deskewed side-by-side | both exact |
| `492637750…jpg` | newspaper photo | ~78/81 (faint grey clues) |
| `492637753…jpg` | newspaper photo | ~88% |
| `492637751…jpg` | newspaper photo | ~65% (perspective straddle) |
| `v4-460px…jpg` | colored grid + pencil occlusion | **not read** (warp shear → filtered, no output) |

13 regression tests pass (`tests/test_pipeline.py`). Imperfect grids are flagged by the
row/col/box validity warnings in `validate.py`; grids with ≥10 conflicts (page frames,
unrecoverable warps) are dropped instead of emitting garbage.

---

## What was tried

### 1. Digit classifier: HOG + SVM on synthetic font renders
**Why:** offline, no API; printed digits suit a feature-based classifier. Trained on
1–9 rendered from system fonts.
**Problems found + fixes:**
- Symbol/emoji/non-Latin fonts poisoned training (mapped 1–9 to identical/garbage
  glyphs). → Name blacklist + per-font *distinctness* check (`digits_are_distinct`).
  Train acc 0.94 → 1.00, fonts 38 → 162.
- Real glyphs misread despite looking correct. Root cause: **warp resolution**. Warping
  to 450 px gave 50 px cells; high-res sources (sample.png ~130 px) were downsampled to
  blobs. → `detect.SIZE` 450 → **900**. Biggest single fix.
- Crisp synthetic renders ≠ low-res/blurry real glyphs. → augmentation adds
  **resolution degradation** (downsample→upsample), blur, noise.
- Bold thick-stroke digits (sample.png) collapsed loops of 6/8/9 → read as 1/3. →
  **stroke-weight jitter** (dilate/erode, kernels 2–4).
- HOG alone can't see whether a loop is open or filled. → feature vector is now
  **HOG + downsampled 14×14 pixels**. Eliminated the last 1/3/6/8/9 confusion on the
  clean grids → sample.png exact.

### 2. Empty-cell detection
- First attempt: per-cell Otsu + central ink ratio. Failed on book scans — uniform/
  textured empty cells and **bleed-through ghosts** read as ink (all 81 "filled").
- Fix: binarize the **whole warped grid once** (global Otsu) so faint ghosts stay
  background, then keep only the **largest interior connected component** per cell
  (`extract_glyph`). One test drives both empty-detection and recognition.

### 3. Grid detection robustness
- Strict `approxPolyDP == 4 points` rejected newspaper grids: ragged borders give
  5–12 points even though the grid is the largest, squarest contour. → `_quad_from_contour`
  uses **convex hull + epsilon search** to 4 corners, with a **minAreaRect** boxPoints
  fallback. All 3 newspaper grids now detect (was 1 undetected, 1 garbage-fallback).

### 4. Grid-line removal
**Why:** skewed grid-line fragments inside cells were read as `1`s on the noisier
newspaper warps. → `grid_lines` isolates long horizontal/vertical runs via 1-D
morphological opening (kernel `SIZE//10`); subtracted before cell extraction. Killed
the false-`1` flood on 753.

### 5. CLAHE contrast boost
**Why:** 750 has faint grey clue digits near the background level; global Otsu missed
them. → CLAHE (`clipLimit=1.5`, 8×8 tiles) before binarize. 750 went from rows 1–2
broken to rows 2–9 perfect. Trade-off: mildly amplifies paper-texture noise on 753.

### 6. Adaptive cell boundaries
**Why:** the linear 4-corner warp divides the outer quad *uniformly*, but under
perspective the real cell lines aren't uniformly spaced → ~1-column drift (worst on
751). → `_ten_boundaries` derives boundaries from detected grid lines: 10 lines → use
directly; 4 (only thick box separators survive) → box-anchored interpolation; else
even split. Gated to lines that span the full warp so a couple of faint interior lines
can't be mistaken for the extent. 751 rows 4/6/9 became exact.

---

### 7. Multi-puzzle pages + frame/garbage suppression
**Why:** new samples include 4-per-page and 2-per-page layouts, a puzzle+solution pair,
and a warped+clean pair. Two problems surfaced:
- A 4-per-page sheet has a rounded **outer page border** that is the largest square
  contour. The old `_dedup` used that frame's large radius for center-proximity, so all
  4 inner grids fell within "half radius" and were suppressed → only the frame survived
  (garbage). → `_dedup` now only merges **similar-size** concentric borders (inner vs
  outer line of one grid), so a big frame can't swallow the smaller distinct grids.
- That keeps the frame itself as an extra candidate. Trying to drop it by "encloses a
  smaller candidate" was **wrong**: a normal grid legitimately encloses its own 3×3
  boxes (detected as squares), so any enclosure rule deletes real grids (broke 751 and
  the warped/clean pair). → Instead, suppress at the **output** level: a frame warped
  across 4 grids yields many row/col/box conflicts (frame = 17, v4 shear = 25) whereas
  real puzzles have ≤6. `cli.MAX_CONFLICTS = 10` drops these without emitting garbage.

Spurious sub-squares (3×3 boxes, junk) are harmless: they warp to near-empty grids and
fail the `MIN_CLUES = 17` filter.

## What was tried and reverted

- **Per-band (top/mid/bottom) x-boundaries** to track tilted vertical lines: ~0 extra
  gain over global box-anchored boundaries; removed for simplicity.
- **Digit-height shape gate** (reject squat components as smudges): dropped the small
  digital-theme digits (height ratio ~0.22 of the cell); removed.
- **Illumination flattening in `binarize`** (divide-by-background, then black-hat) to
  fix `v4`'s colored checkerboard cells: it cleaned v4's binary, but v4 still failed on
  the warp shear (the real blocker), and the flattening erased `492637751`'s faint
  low-contrast digits (clues 24 → 11, dropped below `MIN_CLUES`). Net negative → reverted
  to plain auto-polarity Otsu. v4 needs the warp fix, not better binarization.

---

## What still can be tried

1. **Homography re-warp from grid-line intersections (highest value for 751 AND v4).**
   Detect the full 10×10 line lattice, intersect to get precise inner corners, and
   re-warp from those instead of the contour hull/minAreaRect. Fixes both remaining
   failures:
   - **751**: sub-cell perspective drift makes top-band digits straddle a boundary.
   - **v4**: a **pencil occluding the grid** connects to the border in the line mask, so
     the contour (and its hull and minAreaRect) balloons the bottom-right corner out to
     the pencil → the warp is a sheared parallelogram, unreadable. The lattice lines are
     not affected by the pencil, so intersecting them recovers true corners.
   Isolated to `detect.py`. Risk: must keep the 11 clean-case warps intact.

2. **Straddle-tolerant cell assignment.** Search each digit in a window that overlaps
   neighboring cells, then assign by component centroid. Cheaper than #1, directly
   targets the "digit split across a boundary" error.

3. **CNN digit classifier** trained on real grid-cell crops (or synthetic + real mix).
   Should beat HOG+SVM on faint / ambiguous 3/8/9 in newspaper print. Heavier dep
   (torch) and a labeled set; keep SVM as the offline default.

4. **Per-cell adaptive thresholding** instead of one global Otsu, for grids with uneven
   lighting/shadow gradients across the page. Pair with the largest-component filter to
   stay robust to bleed-through.

5. **Multi-orientation / deskew retry.** If a detected grid yields many validity
   conflicts, retry with a small rotation sweep or alternate binarization and keep the
   lowest-conflict result.

6. **Solver-assisted correction.** Feed the extracted grid to a Sudoku solver; if it's
   unsolvable, use the conflict cells to flag/retry just those cells (e.g. second-best
   SVM class). Turns the validity check into active error correction.

7. **Confidence output.** Expose SVM `decision_function` margins so low-confidence cells
   can be surfaced (`?`) or routed to a fallback — useful signal for the photo cases.
