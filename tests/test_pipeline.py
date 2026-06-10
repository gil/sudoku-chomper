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


@pytest.mark.parametrize("name", [
    "sample.png",
    "Screenshot 2026-06-10 at 20.11.58.png",
])
def test_output_is_valid_sudoku_shape(name):
    for puzzle in extract(s(name)):
        assert len(puzzle) == 81
        assert set(puzzle) <= set("0123456789")
