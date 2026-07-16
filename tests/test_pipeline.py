"""End-to-end regression tests over the sample images.

Covers the clean digital / printed cases (exact match) and the multi-grid page.
Noisy newspaper photos are a known hard limit and are not asserted here.
"""

import os

import pytest

from sudoku_chomper.cli import extract

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


# Filled-in book pages: only the printed givens should survive, with no flag — the
# filter is always on and auto-selects per grid. Tier 1 (light pencil) and Tier 2
# (blue ink) are handled by the intensity + saturation path. Tier 3 (dark-pen
# handwriting: dirty_03/04) is a known limit — the marks are as dark and achromatic
# as print — and is not asserted.

def test_printed_only_light_pencil():
    """dirty_01 (= old page_018): bold printed givens, faint gray pencil answers.

    r6c7 = printed 9 the pre-quantization-fix path dropped (verified on the scan).
    """
    assert extract(s("dirty_01.jpg")) == [
        "000078002006300007530000060001000020005080600040000900080000074400002300100760000"
    ]


def test_printed_only_pencil_dirty_06():
    """r5c2 = printed 4 the pre-quantization-fix path dropped (verified on the scan)."""
    assert extract(s("dirty_06.jpg")) == [
        "450000006000020004009800120070008000040000090000900070025004700700050000100000069"
    ]


def test_printed_only_pencil_dirty_07():
    """r2c1 = printed 7 and r9c2 = printed 4 the pre-quantization-fix path dropped
    (confirmed by the labeled rescan of the same page, the 000002…jpg sample)."""
    assert extract(s("dirty_07.jpg")) == [
        "000002080705000400020600090500060200000408000009030005080001030006000708040300000"
    ]


def test_printed_only_blue_ink():
    """dirty_05: black printed givens, blue-pen answers — saturation drops the ink.

    r3c8 was a faint blue 5 (saturation 37, below the filter) leaking through as a
    misread 4; the intensity split now catches it.
    """
    assert extract(s("dirty_05.png")) == [
        "500040007082900000040600100018500000600020003000003690001009060000005480200010005"
    ]


def test_printed_only_pencil_puzzle_8():
    """Filled book page (Puzzle 8), heavy pencil; filename is the labeled truth.

    Hardest intensity case so far: the darkest pencil glyph sits 0.5 gray levels
    above the lightest printed given.
    """
    assert extract(s("850000001000005306040006900017009000000000000000300870008700040602100000100000035.jpg")) == [
        "850000001000005306040006900017009000000000000000300870008700040602100000100000035"
    ]


def test_printed_only_pencil_puzzle_13():
    """Rescan of the dirty_07 page (Puzzle 13); filename is the labeled truth."""
    assert extract(s("000002080705000400020600090500060200000408000009030005080001030006000708040300000.jpg")) == [
        "000002080705000400020600090500060200000408000009030005080001030006000708040300000"
    ]


def test_printed_only_pencil_puzzle_7():
    """Filled book page (Puzzle 7), faint pencil answers; filename is the labeled truth."""
    assert extract(s("098000004000930008000007000009604000750000041000503600000100000600095000300000480.jpg")) == [
        "098000004000930008000007000009604000750000041000503600000100000600095000300000480"
    ]


def test_printed_only_dark_pen_dirty_02():
    """Dark-pen B&W scan, read exactly by the style filter (truth = the
    train_style.REAL_SAMPLES label). Only works since the saturation cluster split:
    the old absolute cut dropped half the glyphs as chroma fringing before the style
    path could see them."""
    assert extract(s("dirty_02.png")) == [
        "000002503031000008024051000000017065260000047710560000000680290400000830603400000"
    ]


# Tinted (aged) book pages: the tan paper lifts every glyph's mean saturation past any
# absolute colored-ink cut, so saturation is clustered per page (SAT_GAP) instead.
# os* = pencil answers, oos* = red-pen answers; os001/oos001, os002/oos002 and
# os003/oos004 are scans of the same book pages.

def test_tinted_page_pencil_os001():
    assert extract(s("os001.jpg")) == [
        "026000810300708006400050007050107090003905100040302050100030002500204009038000460"
    ]


def test_tinted_page_red_pen_oos001():
    """Same page as os001 answered in red pen — saturation must drop the pen but keep
    the print, which sits well above 50 mean saturation on this paper."""
    assert extract(s("oos001.jpg")) == [
        "026000810300708006400050007050107090003905100040302050100030002500204009038000460"
    ]


def test_tinted_page_pencil_os002():
    assert extract(s("os002.jpg")) == [
        "063002410400508007800103006987000140000030000024000695700201004600309001018400730"
    ]


def test_tinted_page_scan_pairs_agree():
    """Independent scans of one page (pencil vs red pen) must extract identically."""
    assert extract(s("oos002.jpg")) == extract(s("os002.jpg"))
    assert extract(s("oos004.jpg")) == extract(s("os003.jpg"))


def test_tinted_page_unfilled():
    """Clean unfilled page from the same tinted-paper book; the old absolute
    saturation cut dropped all 27 printed givens as colored ink."""
    assert extract(s("not-det.jpg")) == [
        "040100700800050030009000005070001000300060008000400090500000200010030006004008070"
    ]


def test_filled_page_returns_givens_not_solved():
    """A fully filled scan now yields its printed givens (handwriting dropped), not []."""
    assert extract(s("dirty_01.jpg")) == [
        "000078002006300007530000060001000020005080600040000900080000074400002300100760000"
    ]


# Four scans of one tinted book page (Puzzle 28): twins 1/3 are fully answered in
# red ink, twins 2/4 partially in pencil, and twins 2/4 carry stray pencil strokes
# crossing the grid border (the appendage-stripping path in detect). All four must
# reduce to the same printed givens. r4c9 holds only handwriting in every scan (a
# red 7 in twins 1/3; in twins 2/4 the solver wrote a wrong pencil 5 there), so it
# is not a printed given despite the red version looking print-dark.

TWINS_GIVENS = "030050000900301800004000070080024900010000030007960040050000300006405001000070060"


@pytest.mark.parametrize("name", ["twin-1.png", "twin-2.png", "twin-3.jpg", "twin-4.jpg"])
def test_twin_scans_agree(name):
    assert extract(s(name)) == [TWINS_GIVENS]


# Tinted book pages ("Puzzle N" series) answered in pencil, dark ballpoint, or blue
# pen — several fully solved. They exercise: the darker-cluster veto in the
# saturation split (dark print on tan paper measures high-saturation and must not be
# dropped as colored ink), the relaxed width gate retried once a grid reads full or
# self-conflicting, the outlier peel in the width clustering (scribbled-over cells),
# and the photometric rescue/veto on the style path. Every extraction below is
# conflict-free and uniquely solvable.

@pytest.mark.parametrize("name,expected", [
    ("why_no_01.jpg", "040509070206000104030000050300907002000040000700108003080000030109000807070206010"),
    ("why_no_02.jpg", "000023000000900004018000005000006030070000020050100000900000860300007000000540000"),
    ("why_no_03.jpg", "002000800030107090400006001056000020000090000070000450800300006090208070001000900"),
    ("why_no_04.jpg", "000000700007008030090600200060400000008000500001009080002005070040900600003000000"),
    ("why_no_05.jpg", "004090000000000860503001020000502900700000004005304000060800503021000000000140700"),
    ("why_no_06.jpg", "000004310200810000700090000100000050063000840090000002000020009000053004054700000"),
    ("why_no_07.jpg", "000970064000402005000050900120006080905000706080300041004030000500207000630014000"),
    ("why_no_08.jpg", "300500840085006002090003006500600490000070000023004007400100080900700230056008001"),
])
def test_tinted_book_page_givens(name, expected):
    assert extract(s(name)) == [expected]


# _style_keep gate behavior (model-free): the stroke-width fusion must only fire when
# the grid really holds two ink sources.

def test_style_keep_uniform_grid_kept_whole():
    """One font, one ink: tight widths + tight scores -> nothing is dropped."""
    from sudoku_chomper.cells import _style_keep

    scores = [-1.2, -1.1, -1.3, -1.15, -1.25, -1.18]
    widths = [6.0, 6.2, 5.9, 6.1, 6.0, 6.05]
    assert _style_keep(scores, widths) == [True] * 6


def test_style_keep_splits_two_ink_sources():
    """Thick print + thin pen with matching style scores -> thin cluster dropped."""
    from sudoku_chomper.cells import _style_keep

    scores = [-1.0, -1.1, -0.9, 0.8, 0.9, 1.0]
    widths = [6.0, 6.2, 5.8, 2.0, 2.2, 1.9]
    assert _style_keep(scores, widths) == [True, True, True, False, False, False]
