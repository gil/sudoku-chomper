"""Train the printed-vs-handwritten style classifier (Option B prototype).

Tier-3 scans (dirty_02/03/04) are black-and-white: printed givens and handwritten
answers share the same darkness and hue, so intensity/saturation can't tell them
apart — only glyph *shape* can. This trains a binary SVM:

    class 0 = printed   -> synthetic font renders (same pipeline as train.py)
    class 1 = handwritten -> MNIST digits 1-9

Both go through the shared normalize_glyph + hog_features pipeline so live cell crops
land in the same feature space. Run once:

    python -m sudoku_chomper.train_style
"""

from __future__ import annotations

import os
import random

import cv2
import numpy as np

from .recognize import hog_features, normalize_glyph
from .train import RENDER, augment, digits_are_distinct, find_fonts, render_digit

import sklearn  # noqa: F401  (ensure available before the long font/MNIST work)
from PIL import ImageFont

STYLE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "models", "style_svm.joblib"
)
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "sample")

AUG_PER_FONT = 12  # printed samples per font per digit (keeps classes balanceable)
MNIST_PER_CLASS = 800  # handwritten samples per digit 1-9
AUG_PER_REAL = 12  # variants per real labeled glyph

# Real filled-in scans + their printed-givens strings. These anchor the classifier in
# the actual print/handwriting domain (B&W book pages), which MNIST alone misses: a
# nonzero digit marks a printed given, "0" a handwritten answer. Add more pages here as
# they are labeled — separation and precision scale directly with this set.
REAL_SAMPLES = {
    "dirty_02.png": "000002503031000008024051000000017065260000047710560000000680290400000830603400000",
    "dirty_03.png": "800070030000005600004003200000050070100306008020040000006100400005800000070020003",
    "dirty_04.png": "003000004004800030970003500040506900000030000002109080001200067090008400300000100",
}


def build_printed(rng: random.Random) -> list[np.ndarray]:
    fonts = find_fonts()
    if not fonts:
        raise RuntimeError("No usable system fonts found for training.")
    X: list[np.ndarray] = []
    loaded = 0
    for path in fonts:
        try:
            font = ImageFont.truetype(path, RENDER - 8)
        except Exception:
            continue
        if digits_are_distinct(font) is None:
            continue
        loaded += 1
        for d in "123456789":
            base = render_digit(font, d)
            for s in [base] + [augment(base, rng) for _ in range(AUG_PER_FONT)]:
                canvas = normalize_glyph(s)
                if canvas is not None:
                    X.append(hog_features(canvas))
    print(f"printed: {loaded} fonts -> {len(X)} samples")
    return X


def build_handwritten() -> list[np.ndarray]:
    from sklearn.datasets import fetch_openml

    print("fetching MNIST (cached after first run)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    imgs = mnist.data.reshape(-1, 28, 28).astype(np.uint8)
    labels = mnist.target.astype(int)

    rng = np.random.default_rng(1234)
    X: list[np.ndarray] = []
    for d in range(1, 10):  # sudoku has no 0; match the printed class distribution
        idx = np.where(labels == d)[0]
        rng.shuffle(idx)
        for i in idx[:MNIST_PER_CLASS]:
            # MNIST is white-on-black grayscale; binarize to an ink mask, then run the
            # same normalization the live pipeline uses.
            _, mask = cv2.threshold(imgs[i], 40, 255, cv2.THRESH_BINARY)
            canvas = normalize_glyph(mask)
            if canvas is not None:
                X.append(hog_features(canvas))
    print(f"handwritten: -> {len(X)} samples")
    return X


def build_real(rng: random.Random) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Labeled glyphs from real filled-in scans, augmented. Returns (printed, hand)."""
    from .detect import find_grids, load_image
    from .cells import _ten_boundaries, grid_lines, INSET
    from .recognize import binarize, extract_glyph

    Xp: list[np.ndarray] = []
    Xh: list[np.ndarray] = []
    for name, truth in REAL_SAMPLES.items():
        path = os.path.join(SAMPLE_DIR, name)
        if not os.path.exists(path):
            print(f"  (skip missing {name})")
            continue
        gray = find_grids(load_image(path))[0]
        eq = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(gray)
        binary = binarize(eq)
        lines = grid_lines(binary)
        clean = cv2.subtract(binary, lines)
        bx = _ten_boundaries(lines.sum(axis=0))
        by = _ten_boundaries(lines.sum(axis=1))
        for row in range(9):
            for col in range(9):
                ch, cw = by[row + 1] - by[row], bx[col + 1] - bx[col]
                y0 = int(by[row] + ch * INSET); y1 = int(by[row + 1] - ch * INSET)
                x0 = int(bx[col] + cw * INSET); x1 = int(bx[col + 1] - cw * INSET)
                g = extract_glyph(clean[y0:y1, x0:x1])
                if g is None:
                    continue
                bucket = Xp if truth[row * 9 + col] != "0" else Xh
                for v in [g] + [augment(g, rng) for _ in range(AUG_PER_REAL)]:
                    c = normalize_glyph(v)
                    if c is not None:
                        bucket.append(hog_features(c))
    print(f"real: -> {len(Xp)} printed / {len(Xh)} handwritten")
    return Xp, Xh


def main() -> None:
    from sklearn.svm import SVC

    rng = random.Random(1234)
    np.random.seed(1234)
    Xp = build_printed(rng)
    Xh = build_handwritten()
    rp, rh = build_real(rng)
    Xp += rp
    Xh += rh

    X = np.array(Xp + Xh)
    y = np.array([0] * len(Xp) + [1] * len(Xh))
    print(f"total: {len(X)} samples ({len(Xp)} printed / {len(Xh)} handwritten)")

    clf = SVC(kernel="rbf", C=10, gamma="scale")
    clf.fit(X, y)
    print(f"Train accuracy: {clf.score(X, y):.4f}")

    import joblib

    os.makedirs(os.path.dirname(os.path.abspath(STYLE_MODEL_PATH)), exist_ok=True)
    joblib.dump(clf, STYLE_MODEL_PATH)
    print(f"Saved -> {os.path.abspath(STYLE_MODEL_PATH)}")


if __name__ == "__main__":
    main()
