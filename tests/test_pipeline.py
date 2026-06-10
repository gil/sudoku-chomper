"""End-to-end regression tests over the sample images.

Covers the clean digital / printed cases (exact match) and the multi-grid page.
Noisy newspaper photos are a known hard limit and are not asserted here.
"""

import os

import pytest

from sudoku_ocr.cli import extract

SAMPLES = os.path.join(os.path.dirname(__file__), os.pardir, "sample")


def s(name: str) -> str:
    return os.path.join(SAMPLES, name)


def test_digital_screenshot_exact():
    assert extract(s("sample.png")) == [
        "750000000003400006601200040490000000002835900000000052040006703300009400000000068"
    ]


def test_printed_book_scan_17():
    assert extract(s("Screenshot 2026-06-10 at 20.11.58.png")) == [
        "000000015049000000200301700800200090090000070070006001004905007000000540610000000"
    ]


def test_printed_book_scan_20():
    assert extract(s("Screenshot 2026-06-10 at 20.12.10.png")) == [
        "000700060002080004040002030800000500090603080006000003020500090600040200080001000"
    ]


def test_two_puzzles_on_one_page():
    """Page 292 holds puzzles 339 and 340 stacked — both must be read, in order."""
    assert extract(s("Screenshot 2026-06-10 at 20.13.58.png")) == [
        "090060080150090047000205000007109300630000014005608700000301000970050068080070020",
        "004520000150000046000067500031800000600000003000005210003180000520000067000052400",
    ]


def test_photo_websudoku_printout():
    assert extract(s("0-sudoku.jpg")) == [
        "040063057060010403203700086030090008000802000800070090510006809409080010380150070"
    ]


def test_rotated_newspaper_clipping():
    """~30° rotated torn clipping — quad detection + warp must handle the rotation."""
    assert extract(s("Sudoku.jpg")) == [
        "030070506960040000004016270020037004000020000700460010019650300000080091602090050"
    ]


def test_four_puzzles_per_page_colored():
    assert extract(s("30be7414e478fe4825d5af81cc47fa14.jpg")) == [
        "070020046060000890200800715084097000710000059000130480697002008058000060430080070",
        "004050000900734600003021049035090480090000030076010920310970200009182003000060100",
        "380900205000008730060300980000003501910507023703100000035001090074600000801002067",
        "806010000003064090900000816080396000709040309000572080521000004030750200000020105",
    ]


def test_four_puzzles_per_page_with_outer_frame():
    """Page has a rounded outer border enclosing 4 grids; the frame must be filtered
    (high conflict count) and exactly the 4 puzzles returned."""
    result = extract(s("sudoku-evil-5.jpg"))
    assert len(result) == 4
    assert result[0] == "026000708190060040740900001001004670004610009609800010007056004010089200800400900"
    assert result[1] == "500087040900050706270004030026005400089400005400370800004760050007008604050900087"


def test_puzzle_and_solution_side_by_side():
    """Left grid is the puzzle, right grid is the filled solution — only the puzzle."""
    assert extract(s("sudoku_solved.png")) == [
        "000208006006900040030050000890002004007000600500700091000070010020006700100405000"
    ]


def test_warped_and_deskewed_side_by_side():
    """Composite of a warped photo + its clean version; both read to the same puzzle."""
    puzzle = "004700009078000010950020000400052000001304900000890006000010097090000650700003400"
    assert extract(s("sudoku-warped-example.png")) == [puzzle, puzzle]


def test_two_puzzles_per_page_first_exact():
    result = extract(s("s-l1200.png"))
    assert len(result) == 2
    assert result[0] == "090030008750046000020915430468000050001000200030000189019573020000120095500080070"


@pytest.mark.parametrize("name", [
    "sample.png",
    "Screenshot 2026-06-10 at 20.11.58.png",
])
def test_output_is_valid_sudoku_shape(name):
    for puzzle in extract(s(name)):
        assert len(puzzle) == 81
        assert set(puzzle) <= set("0123456789")
