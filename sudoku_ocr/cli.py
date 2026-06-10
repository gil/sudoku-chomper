"""CLI: extract Sudoku puzzle(s) from an image as 81-char strings."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import cv2
import numpy as np

from . import validate
from .cells import iter_cells
from .detect import find_grids, load_image
from .recognize import predict_glyph

MIN_CLUES = 17  # fewest givens a proper Sudoku can have; filters false-positive grids


def grid_to_string(warped: np.ndarray, debug_dir: str | None = None, idx: int = 0) -> str:
    digits = ["0"] * 81
    for i, glyph in iter_cells(warped):
        if glyph is not None:
            d = predict_glyph(glyph)
            if d:
                digits[i] = str(d)
            if debug_dir is not None:
                cv2.imwrite(os.path.join(debug_dir, f"grid{idx}_cell{i:02d}_{digits[i]}.png"), glyph)
    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, f"grid{idx}_warp.png"), warped)
    return "".join(digits)


def extract(path: str, include_all: bool = False, debug: bool = False) -> list[str]:
    image = load_image(path)
    grids = find_grids(image)

    debug_dir = None
    if debug:
        debug_dir = tempfile.mkdtemp(prefix="sudoku_ocr_")
        print(f"# debug crops -> {debug_dir}", file=sys.stderr)

    results: list[str] = []
    for idx, warped in enumerate(grids):
        puzzle = grid_to_string(warped, debug_dir, idx)
        n = validate.filled_count(puzzle)
        if n < MIN_CLUES:
            continue  # not a plausible Sudoku grid
        if n == 81 and not include_all:
            continue  # solved grid (e.g. printed solution) — puzzles only
        for msg in validate.conflicts(puzzle):
            print(f"# warning [{path}]: {msg} (possible OCR misread)", file=sys.stderr)
        results.append(puzzle)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sudoku-ocr",
        description="Extract Sudoku puzzle(s) from an image as 81-char strings.",
    )
    parser.add_argument("image", help="path to the image file")
    parser.add_argument("--all", action="store_true", help="also print fully-solved grids")
    parser.add_argument("--debug", action="store_true", help="dump warped grids / cell crops")
    args = parser.parse_args(argv)

    results = extract(args.image, include_all=args.all, debug=args.debug)
    if not results:
        print("# no Sudoku grid detected", file=sys.stderr)
        return 1
    for line in results:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
