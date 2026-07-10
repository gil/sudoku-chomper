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

23 regression tests pass (`tests/test_pipeline.py`). Imperfect grids are flagged by the
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

---

# Printed vs handwritten: keeping only the printed givens

Filled-in book scans (the `dirty_*` and labeled puzzle samples) carry the original
**printed givens** plus a user's **handwritten answers**. Goal: extract only the printed
givens. This is **always on — no flags** (the old `--printed-only` / `--use-style` flags
were removed 2026-06; the filter auto-selects per grid). Glyph extraction + digit OCR are
already correct on the printed givens (dirty_02 32/32, dirty_03 24/24, dirty_04 27/27);
the only problem is the **print-vs-handwriting decision**.

## Three difficulty tiers (by the signal that separates the two)

| Tier | Samples | Handwriting vs print | Separating signal | Status |
|---|---|---|---|---|
| 1 | `dirty_01/06/07`, Puzzle 7/8/13 | faint grey **pencil** vs bold print | grayscale **intensity** | works |
| 2 | `dirty_05` | **blue/colored ink** vs black print | HSV **saturation** | works |
| 3 | `dirty_02/03/04` | **dark pen**, B&W scan — same darkness *and* hue | glyph **shape** only | auto-selected, in-sample only |

## How the filter is selected (`cells._select_keep`)

The two regimes **conflict** (a B&W shape filter wrongly drops pencil givens; the
intensity split false-fires on clean print), so exactly one is chosen per grid:

1. **Handwriting detector = the style model's stroke-width gate.** `_style_keep` only
   drops cells when two ink sources are actually present (see gates below). If it drops
   **nothing**, the grid is clean / all-printed → **keep every glyph**. This is what
   protects clean grids: the intensity split alone *false-fires* on printed pages with
   tonal variation (`s-l1200`, the 4-up colored sheet, solution grids all show
   intensity-drops with style-drops = 0), so a lone intensity split is never trusted.
2. **Handwriting present → pick the accurate filter.** Use intensity/saturation
   (`_printed_mask`) by default; switch to the style mask only when it drops **far more**
   than intensity (`style_drops ≥ 1.5 × intensity_drops` **and** `≥ 10` more cells) — the
   signature of a dark-pen page where intensity stayed flat. On pencil pages the two drop
   counts are comparable (intensity wins, no leak); on `dirty_03/04` style drops ~50 vs
   intensity ~1–23 → style is selected automatically.
3. **Whole-grid safety net.** If the auto-selected style filter blanks a grid (drops
   below `MIN_CLUES`), `cli.extract` retries it pinned to intensity (`force_intensity`).

Constants `STYLE_OVER_INTENSITY` / `STYLE_DROP_MARGIN` in `cells.py` are the selector
thresholds, alongside the existing gate constants.

## What was built

- **Intensity + saturation path** (`_printed_mask`, Tiers 1–2). Each glyph is scored on
  the **pre-CLAHE** warp (CLAHE destroys absolute intensity):
  - **Saturation** filter drops colored ink. This is a per-page cluster split, not an
    absolute cut: an Otsu split over the glyph saturations drops the high cluster only
    when the cluster gap clears `SAT_GAP` and the high cluster really is colored
    (`SAT_THRESH`). Tinted (aged) paper lifts every glyph's mean saturation together.
    Black print on the tan `os*`/`oos*` book pages measures 55–102, above colored ink
    on white paper, so the old absolute `> 50` cut blanked those grids entirely
    (fixed 2026-07).
  - **Adaptive intensity split**: 1-D Otsu over the grid's glyph intensities, drop the
    light (pencil) cluster — only when the dark/light cluster gap clears `PENCIL_GAP`,
    so a fully-printed grid is kept whole.
  - **Quantization bug (fixed 2026-06):** Otsu ran on uint8-truncated intensities but
    the keep test compared the floats, so a printed given landing on the integer
    threshold (e.g. 50.6 vs thr 50) was dropped. Two labeled rescans (Puzzle 8 / 13
    book pages) exposed it; fixing it recovered **4 dropped givens across
    dirty_01/06/07** that the old regression strings had wrongly locked in as
    handwritten, and the now-honest cluster gap also caught dirty_05's one residual
    leak (a faint blue 5, saturation 37, misread as a given 4). All split decisions
    now happen in the quantized domain.
  Needs the BGR warp, so `detect.find_grids(image, return_color=True)` carries a color
  crop on the same corners.
- **Style path** (`_style_keep`, Tier 3, also the handwriting *detector* for all tiers).
  A binary **printed-vs-handwritten SVM** (`train_style.py` → `models/style_svm.joblib`)
  over the shared HOG features:
  - printed class = synthetic font renders (reuses `train.py`),
  - handwritten class = MNIST 1–9 **+ real labeled glyphs** from `REAL_SAMPLES`
    (the `dirty_02/03/04` ground-truth strings, auto-extracted and augmented).
  `recognize.style_score` returns a signed margin (`<0` printed, `>0` handwritten);
  `_style_keep` fuses it with **median stroke width** (`recognize.stroke_width`,
  2× median distance-transform over the ink — print and pen come from different ink
  sources, so widths form two clusters). The split fires only when two gates pass:
  width-cluster ratio ≥ `SW_RATIO_GATE` (1.38; printed-only grids measure ≤ 1.31,
  mixed ≥ 1.45) **and** the thin cluster style-scores ≥ `STYLE_AGREE` (0.8) more
  handwritten (clean grids ≤ +0.47, handwriting-bearing ≥ +1.44 — kills false fires
  on warped/JPEG-noisy print where width alone wobbles). When fired, glyphs split by
  2-means on `z(width) − z(style_score)` — the z-scoring self-centers per grid, which
  fixes the per-scan score offset — intersected with the largest-gap score cut. These
  gates are exactly why the gate doubles as the handwriting detector in `_select_keep`.
- **Why shape-only on B&W:** the style path *skips* intensity/saturation, because on a
  B&W JPEG the bold print edges pick up **chroma fringing** (saturation ≈ 80 > 50) and
  the saturation filter would wrongly drop the printed givens.
- **Style model required.** With no `style_svm.joblib`, `_select_keep` can't run the
  detector and falls back to intensity-only (warns on stderr) — which false-fires on
  clean grids. Setup builds it; the Docker image copies the host-trained models
  (in-container training used the container's fonts and misread noisy scans,
  so since 2026-07 the build requires pre-trained models).
- **Notice**: when filtering drops any cell, `cli.extract` prints a
  `# note … dropped N handwritten cell(s) … (<path> filter)` to stderr (via the `stats`
  dict threaded through `iter_cells`, which also records the selected path). Output is
  unchanged; it just flags that filtering happened.

## What was measured (style head)

- **MNIST-only** handwritten class: separation ≈ **0.5**, distributions overlap heavily
  (real B&W book glyphs look more "printed" than MNIST). Not usable; also false-split
  clean Tier-1/2 grids. → had to add real samples and gate the regimes.
- **+ real labeled glyphs, in-sample** (model has seen that scan's hand): dirty_02/03/04
  all **exact**.
- **+ real labeled glyphs, leave-one-image-out** (test scan unseen): **recall is perfect**
  (all givens kept) but it **keeps ~all handwriting too** — on an unseen scan the scores
  don't form a clean gap, so the largest-gap cut no-ops. With only 3 scans the
  classifier **overfits per-scan and does not generalize**.
- **Stroke-width fusion (current).** Probing per-glyph features against the
  dirty_02/03/04 ground truth: **median stroke width** is by far the strongest single
  signal (per-grid threshold separability 1.00 / 0.82 / 0.99 — print is thicker than
  ballpoint), beating glyph height, area, centroid, stroke-width variance, and the
  digit-SVM margin. It needs no training, so it generalizes by construction. Fused
  with the style score (see above), **leave-one-image-out**: recall **1.000** on all
  three scans, leak 2/49, 19/57, 0/54 (dirty_03's leak is its thick-pen overwrites).
  In-sample (shipped model): dirty_02/03/04 all **exact**, all handwriting dropped.
- **Auto-selection makes the style path safe to leave always on.** The two gates make
  `_style_keep` byte-identical to no-filter on all clean printed samples (it stays
  silent), and `_select_keep` only *applies* the style mask when it dominates intensity.
  Earlier, the score-only largest-gap cut silently **corrupted printed pages** (false-split
  `0-sudoku` and 3 of 4 grids on the 4-up sheet); the gates plus the selector remove that.
- Pencil/ink pages (`dirty_01/05/06/07`, Puzzle 7/8/13) trip the gates correctly
  out-of-sample, but the *style* split alone leaks 1–6 cells — which is why the selector
  keeps them on the intensity/saturation path (comparable drop counts → intensity wins).

**Bugs fixed along the way**: `_style_keep` uint8 quantization placed the cut at a cluster
edge (dropped 6 givens) → largest-gap split; saturation filter backfiring on B&W (the
regime split); the intensity split false-firing on clean print → gated behind the style
detector in `_select_keep` (2026-06).

## What still can be tried (style head)

1. **More labeled scans — still the biggest lever for the residual leak.** ~**20–40**
   filled pages spanning handwriting styles / books / scanners, labeled via their
   printed-givens strings (add to `REAL_SAMPLES`; extraction is automatic). Recall is
   solved by the width fusion; more data is what shrinks dirty_03-style leak (19/57)
   on unseen scans, and would let Tier-3 be asserted out-of-sample.
2. **CNN / better features** for the style head once a real labeled set exists — HOG was
   chosen to reuse the digit pipeline, but shape discrimination of print vs hand may want
   richer features. Keep SVM as the offline default.
3. **Same-width pen blind spot.** The `SW_RATIO_GATE` assumes print is thicker than the
   pen; a marker matching the print's stroke weight keeps the gate shut and nothing
   splits (safe failure: output includes handwriting, conflicts warn). Only the style
   score can cover this — needs the data from #1.
4. **dirty_03 scribbles / overwrites.** Where handwriting overwrites a printed cell the
   connected components merge — likely unrecoverable without stroke-level separation
   (stroke-width transform / inpainting). Treat as a known hard limit. Its 19-cell LOO
   leak is dominated by these thick overwrite blobs.
5. **Selector thresholds from more data.** `STYLE_OVER_INTENSITY` / `STYLE_DROP_MARGIN`
   were set from the handful of bundled pages; `dirty_02` (intensity 54 vs style 49)
   stays on intensity and remains a known limit. More labeled mixed pages would let the
   intensity-vs-style switch be tuned (or learned) rather than hand-set.

## Coverage today (all auto-selected, no flags)

| Sample | Tier | Selected path | Result |
|---|---|---|---|
| `dirty_01/06/07` | 1 (pencil) | intensity | exact (post quantization fix) |
| `dirty_05` | 2 (blue ink) | intensity | exact (post quantization fix) |
| `098…480.jpg` (Puzzle 7) | 1 (pencil) | intensity | **exact** |
| `850…035.jpg` (Puzzle 8) | 1 (heavy pencil) | intensity | **exact** (label corrected: corner is a printed 5) |
| `000…030…000.jpg` (Puzzle 13, dirty_07 rescan) | 1 (pencil) | intensity | **exact** |
| `dirty_02` | 3 (dark pen) | intensity | not separable — known limit |
| `dirty_03/04` | 3 (dark pen) | style | in-sample exact; LOO recall 1.000, leak 19/0 cells |

Heavy pencil pressed to near-print darkness (Puzzle 8's corner cells, intensity gap of
only 0.5 gray levels at the boundary) is the practical floor of the intensity signal —
beyond it, only shape/stroke fusion can help.

Two mislabeled duplicate samples were dropped (2026-06): `…034…` (a handwritten 4
labeled as a given) and `…036` (a printed 5 labeled 6) were byte-identical to the
correctly-named `…030…` / `…035` files.

Tier-1/2 regression tests (`dirty_01/05/06/07`, Puzzle 7/8/13) are in
`tests/test_pipeline.py`, now called with no flag. Tier-3 end-to-end is not asserted (the
shipped model is in-sample there; would just lock the model file), but the gate logic
itself is unit-tested model-free (`test_style_keep_uniform_grid_kept_whole`,
`test_style_keep_splits_two_ink_sources`).
