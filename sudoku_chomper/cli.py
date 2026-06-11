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
MAX_CONFLICTS = 10  # above this a "grid" is a page frame / unrecoverable warp, not a puzzle


def grid_to_string(warped: np.ndarray, debug_dir: str | None = None, idx: int = 0,
                   printed_only: bool = False, color_warped: np.ndarray | None = None,
                   use_style: bool = False, stats: dict | None = None) -> str:
    digits = ["0"] * 81
    for i, glyph in iter_cells(warped, printed_only=printed_only,
                               color_warped=color_warped, use_style=use_style, stats=stats):
        if glyph is not None:
            d = predict_glyph(glyph)
            if d:
                digits[i] = str(d)
            if debug_dir is not None:
                cv2.imwrite(os.path.join(debug_dir, f"grid{idx}_cell{i:02d}_{digits[i]}.png"), glyph)
    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, f"grid{idx}_warp.png"), warped)
    return "".join(digits)


def extract(path: str, include_all: bool = False, debug: bool = False,
            printed_only: bool = False, use_style: bool = False) -> list[str]:
    printed_only = printed_only or use_style  # style filtering needs the printed-only path
    image = load_image(path)
    grids = find_grids(image, return_color=printed_only)

    debug_dir = None
    if debug:
        debug_dir = tempfile.mkdtemp(prefix="sudoku_chomper_")
        print(f"# debug crops -> {debug_dir}", file=sys.stderr)

    results: list[str] = []
    for idx, grid in enumerate(grids):
        warped, color_warped = grid if printed_only else (grid, None)
        stats: dict = {}
        puzzle = grid_to_string(warped, debug_dir, idx, printed_only, color_warped,
                                use_style, stats)
        if stats.get("style_missing"):
            print(f"# warning [{path}]: --use-style but no style model; "
                  "fell back to intensity/saturation. Run: "
                  "python -m sudoku_chomper.train_style", file=sys.stderr)
        if stats.get("filtered"):
            print(f"# note [{path}]: dropped {stats['filtered']} handwritten cell(s) "
                  f"from grid {idx} (--printed-only)", file=sys.stderr)
        n = validate.filled_count(puzzle)
        if n < MIN_CLUES:
            continue  # not a plausible Sudoku grid
        if n == 81 and not include_all:
            continue  # solved grid (e.g. printed solution) — puzzles only
        conflicts = validate.conflicts(puzzle)
        if len(conflicts) >= MAX_CONFLICTS:
            continue  # not a real puzzle (page frame / unrecoverable warp), not OCR noise
        for msg in conflicts:
            print(f"# warning [{path}]: {msg} (possible OCR misread)", file=sys.stderr)
        results.append(puzzle)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sudoku-chomper",
        description="Extract Sudoku puzzle(s) from an image as 81-char strings.",
    )
    parser.add_argument("image", help="path to the image file")
    parser.add_argument("--all", action="store_true", help="also print fully-solved grids")
    parser.add_argument("--debug", action="store_true", help="dump warped grids / cell crops")
    parser.add_argument("--printed-only", action="store_true",
                        help="keep only printed givens, ignore handwritten answers")
    parser.add_argument("--use-style", action="store_true",
                        help="(experimental) also drop dark handwriting by glyph shape "
                             "via the style model; implies --printed-only")
    args = parser.parse_args(argv)

    results = extract(args.image, include_all=args.all, debug=args.debug,
                      printed_only=args.printed_only or args.use_style,
                      use_style=args.use_style)
    if not results:
        print("# no Sudoku grid detected", file=sys.stderr)
        return 1
    for line in results:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
