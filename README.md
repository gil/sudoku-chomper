# sudoku-chomper

Offline CLI that extracts Sudoku puzzle(s) from an image and prints each as the
standard 81-character string (row-major, `0` = empty cell).

```
750000000003400006601200040490000000002835900000000052040006703300009400000000068
```

Pure local computer vision + OCR — no network, no API. Grid detection with OpenCV,
printed-digit recognition with a HOG + SVM classifier trained on synthetic font
renders.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m sudoku_chomper.train        # builds models/digit_svm.joblib (run once)
python -m sudoku_chomper.train_style  # builds models/style_svm.joblib (run once)
```

Both models are built once and cached. `digit_svm` recognizes the 1–9 glyphs;
`style_svm` is the printed-vs-handwritten detector that gates handwriting removal
(see Usage). Without it the tool falls back to intensity/saturation only and warns.

## Usage

```bash
python -m sudoku_chomper IMAGE [--all] [--debug]
```

- Prints one 81-char line per detected **unsolved** grid (top-to-bottom,
  left-to-right). An image may contain more than one puzzle.
- Only **printed givens** are returned — handwritten / penciled-in answers are
  dropped automatically. Per grid, the style model first decides whether the grid
  even holds handwriting: clean / all-printed grids keep every glyph. When handwriting
  is present, intensity + saturation drops pencil and colored ink, and the style model's
  shape cue handles black-and-white scans where dark handwriting matches print in both
  tone and hue.
- `--all` — also print fully-filled grids (e.g. printed solution grids), normally
  suppressed.
- `--debug` — dump warped grids and per-cell crops to a temp dir for tuning.

A row/column/box validity check runs on each result; conflicts (likely OCR
misreads) are reported on stderr as `# warning ...` without blocking output.

## Docker

The image copies your locally trained models instead of training its own, so
container detection matches the host exactly. (Training inside the image used the
container's fonts, which misread noisy scans the host models got right.)

```bash
# Train the models first (once, see Setup); the build fails without them:
python -m sudoku_chomper.train && python -m sudoku_chomper.train_style

docker build --progress=plain --no-cache -t sudoku-chomper .

# Mount the folder holding your images, then pass a container-side path:
docker run --rm -v "$PWD/sample:/data" sudoku-chomper /data/sample.png
```

The image's `ENTRYPOINT` is the CLI, so anything after the image name is passed
straight through (`IMAGE [--all] [--debug]`).

## How it works

1. `detect.py` — locate each 9×9 grid (convex-hull → 4-corner quad, with a rotated
   bounding-box fallback so ragged newspaper borders still register) and
   perspective-correct it to a 900 px square.
2. `cells.py` — CLAHE contrast boost, global Otsu binarize, morphological grid-line
   removal, cell boundaries from the detected grid lines (box-anchored when only the
   thick separators survive, else an even split), then isolate each cell's largest
   interior component (empty cells yield none). Finally drop handwritten answers: the
   style model's stroke-width gate detects whether the grid mixes two ink sources — if
   not, every glyph is kept; if so, the more accurate of two filters is selected
   (intensity/saturation for pencil & colored ink, the style shape model for dark B&W
   handwriting).
3. `recognize.py` — classify the surviving printed glyphs (1–9) with the digit SVM on
   HOG + downsampled pixel features; the style SVM (`style_score`, `stroke_width`)
   supplies the handwriting cues used in step 2.
4. `validate.py` — flag row/column/box duplicate conflicts.
5. `cli.py` — assemble strings, filter solved grids, order, print; retries a grid with
   intensity-only filtering if the style filter blanks it.

## Accuracy on the bundled samples

- **Digital screenshot, printed book scans, two-puzzle page** — exact.
- **Filled-in book pages** — only the printed givens are returned; light-pencil and
  colored-ink answers are dropped exactly, including on tinted (aged) paper where the
  page tint lifts every glyph's saturation. Dark-pen B&W handwriting is the known hard
  limit (as dark and achromatic as print).
- **Newspaper photos** — all grids detected and mostly read; faint print, perspective
  skew, and 3/8/9 ambiguity cause occasional cell errors, surfaced by the validity
  warnings.
