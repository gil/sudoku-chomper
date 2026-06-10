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
python -m sudoku_chomper.train      # builds models/digit_svm.joblib (run once)
```

## Usage

```bash
python -m sudoku_chomper IMAGE [--all] [--debug]
```

- Prints one 81-char line per detected **unsolved** grid (top-to-bottom,
  left-to-right). An image may contain more than one puzzle.
- `--all` — also print fully-filled grids (e.g. printed solution grids), normally
  suppressed.
- `--debug` — dump warped grids and per-cell crops to a temp dir for tuning.

A row/column/box validity check runs on each result; conflicts (likely OCR
misreads) are reported on stderr as `# warning ...` without blocking output.

## Docker

No local Python/OpenCV needed. The image installs the dependencies and bakes the
trained digit model in at build time (using fonts installed in the container).

```bash
docker build -t sudoku-chomper .

# Mount the folder holding your images, then pass a container-side path:
docker run --rm -v "$PWD/sample:/data" sudoku-chomper /data/sample.png
docker run --rm -v "$PWD/sample:/data" sudoku-chomper /data/sample.png --all
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
   interior component (empty cells yield none).
3. `recognize.py` — classify non-empty cells (1–9) with the SVM on HOG + downsampled
   pixel features.
4. `validate.py` — flag row/column/box duplicate conflicts.
5. `cli.py` — assemble strings, filter solved grids, order, print.

## Accuracy on the bundled samples

- **Digital screenshot, printed book scans, two-puzzle page** — exact.
- **Newspaper photos** — all grids detected and mostly read; faint print, perspective
  skew, and 3/8/9 ambiguity cause occasional cell errors, surfaced by the validity
  warnings.
