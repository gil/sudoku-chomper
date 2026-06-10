# sudoku-ocr

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
python -m sudoku_ocr.train      # builds models/digit_svm.joblib (run once)
```

## Usage

```bash
python -m sudoku_ocr IMAGE [--all] [--debug]
```

- Prints one 81-char line per detected **unsolved** grid (top-to-bottom,
  left-to-right). An image may contain more than one puzzle.
- `--all` — also print fully-filled grids (e.g. printed solution grids), normally
  suppressed.
- `--debug` — dump warped grids and per-cell crops to a temp dir for tuning.

A row/column/box validity check runs on each result; conflicts (likely OCR
misreads) are reported on stderr as `# warning ...` without blocking output.

## How it works

1. `detect.py` — locate and perspective-correct each 9×9 grid.
2. `cells.py` — slice into 81 cells, flag empties by ink ratio.
3. `recognize.py` — classify non-empty cells (1–9) with the trained SVM.
4. `validate.py` — flag duplicate conflicts.
5. `cli.py` — assemble strings, filter solved grids, order, print.
