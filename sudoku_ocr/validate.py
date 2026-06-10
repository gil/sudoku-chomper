"""Sudoku sanity checks used to flag likely OCR misreads."""

from __future__ import annotations


def conflicts(puzzle: str) -> list[str]:
    """Return duplicate-conflict messages for rows/columns/boxes (empty == clean)."""
    msgs: list[str] = []
    grid = [puzzle[i * 9:(i + 1) * 9] for i in range(9)]

    def dup(cells: list[str], where: str) -> None:
        seen = [c for c in cells if c != "0"]
        if len(seen) != len(set(seen)):
            msgs.append(f"duplicate digit in {where}")

    for r in range(9):
        dup(list(grid[r]), f"row {r + 1}")
    for c in range(9):
        dup([grid[r][c] for r in range(9)], f"column {c + 1}")
    for br in range(3):
        for bc in range(3):
            box = [grid[br * 3 + i][bc * 3 + j] for i in range(3) for j in range(3)]
            dup(box, f"box ({br + 1},{bc + 1})")
    return msgs


def filled_count(puzzle: str) -> int:
    return sum(1 for c in puzzle if c != "0")
