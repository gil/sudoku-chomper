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
from .render import render_puzzle

MIN_CLUES = 17  # fewest givens a proper Sudoku can have; filters false-positive grids
MAX_CONFLICTS = 10  # above this a "grid" is a page frame / unrecoverable warp, not a puzzle


def grid_to_string(warped: np.ndarray, color_warped: np.ndarray | None = None,
                   debug_dir: str | None = None, idx: int = 0,
                   stats: dict | None = None, force_intensity: bool = False,
                   assume_handwriting: bool = False) -> str:
    digits = ["0"] * 81
    for i, glyph in iter_cells(warped, color_warped=color_warped, stats=stats,
                               force_intensity=force_intensity,
                               assume_handwriting=assume_handwriting):
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
    grids = find_grids(image, return_color=True)

    debug_dir = None
    if debug:
        debug_dir = tempfile.mkdtemp(prefix="sudoku_chomper_")
        print(f"# debug crops -> {debug_dir}", file=sys.stderr)

    results = _scan(grids, path, include_all, debug_dir, force_intensity=False)
    if not results:
        # The auto-selected style filter can drop a whole grid the intensity/saturation
        # path would have caught; retry pinned to intensity (may include a few
        # handwritten cells, but a recoverable puzzle beats none).
        results = _scan(grids, path, include_all, debug_dir, force_intensity=True)
    return results


def _scan(grids, path: str, include_all: bool, debug_dir: str | None,
          force_intensity: bool) -> list[str]:
    results: list[str] = []
    for idx, (warped, color_warped) in enumerate(grids):
        stats: dict = {}
        puzzle = grid_to_string(warped, color_warped, debug_dir, idx, stats,
                                force_intensity)
        if stats.get("style_missing"):
            print(f"# warning [{path}]: no style model; using intensity/saturation only. "
                  "Run: python -m sudoku_chomper.train_style", file=sys.stderr)
        if _implies_handwriting(puzzle, include_all):
            # No printed puzzle reads full or self-conflicting, so the grid must hold
            # handwriting the standard width gate missed (e.g. thin-pen answers close
            # to the print's stroke width); retry with the relaxed gate.
            retry_stats: dict = {}
            retry = grid_to_string(warped, color_warped, debug_dir, idx, retry_stats,
                                   force_intensity, assume_handwriting=True)
            if not _implies_handwriting(retry, include_all):
                puzzle, stats = retry, retry_stats
        if stats.get("filtered"):
            print(f"# note [{path}]: dropped {stats['filtered']} handwritten cell(s) "
                  f"from grid {idx} ({stats.get('path', 'intensity')} filter)", file=sys.stderr)
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


def _implies_handwriting(puzzle: str, include_all: bool) -> bool:
    """True when the extraction can only be explained by unfiltered handwriting."""
    if validate.filled_count(puzzle) == 81 and not include_all:
        return True
    return len(validate.conflicts(puzzle)) >= MAX_CONFLICTS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sudoku-chomper",
        description="Extract Sudoku puzzle(s) from an image as 81-char strings.",
    )
    parser.add_argument("image", help="path to the image file")
    parser.add_argument("--all", action="store_true", help="also print fully-solved grids")
    parser.add_argument("--debug", action="store_true", help="dump warped grids / cell crops")
    parser.add_argument("--render-image", nargs="?", const="", metavar="OUT",
                        help="also save each extracted puzzle as a clean digital grid "
                             "image (OUT_sudoku001.ext, ...); defaults to the input image path")
    args = parser.parse_args(argv)

    results = extract(args.image, include_all=args.all, debug=args.debug)
    if not results:
        print("# no Sudoku grid detected", file=sys.stderr)
        return 1
    for line in results:
        print(line)
    if args.render_image is not None:
        _render_outputs(args.render_image or args.image, results)
    return 0


def _render_outputs(base: str, results: list[str]) -> None:
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = ".png"
    for i, puzzle in enumerate(results, 1):
        out = f"{stem}_sudoku{i:03d}{ext}"
        cv2.imwrite(out, render_puzzle(puzzle))
        print(f"# rendered -> {out}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
