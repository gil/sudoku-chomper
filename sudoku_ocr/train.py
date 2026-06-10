"""Train the printed-digit SVM on synthetic font renders.

Digits 1–9 are rendered from a range of system fonts with augmentation (scale,
rotation, blur, noise), pushed through the same normalize+HOG pipeline used at
inference time, then fit with an RBF SVM. Run once:

    python -m sudoku_ocr.train
"""

from __future__ import annotations

import glob
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .recognize import MODEL_PATH, hog_features, normalize_glyph

RENDER = 48  # canvas size for the initial font render before normalization

FONT_GLOBS = [
    "/System/Library/Fonts/*.ttf",
    "/System/Library/Fonts/*.ttc",
    "/System/Library/Fonts/Supplemental/*.ttf",
    "/System/Library/Fonts/Supplemental/*.ttc",
    "/Library/Fonts/*.ttf",
    "/Library/Fonts/*.ttc",
    "/usr/share/fonts/**/*.ttf",
]

# Fonts whose glyph set isn't plain Latin digits (symbols, emoji, dingbats, scripts
# that map 1–9 to unrelated shapes). These poison the classifier.
FONT_NAME_BLOCK = (
    "emoji", "symbol", "dingbat", "webding", "wingding", "icon", "ornament",
    "apple braille", "bodoni ornaments", "notonastaliq", "kohinoor", "devanagari",
    "gujarati", "tamil", "telugu", "kannada", "bangla", "gurmukhi", "oriya",
    "malayalam", "sinhala", "myanmar", "khmer", "lao", "zapf", "music",
)


def find_fonts() -> list[str]:
    paths: list[str] = []
    for pat in FONT_GLOBS:
        paths.extend(glob.glob(pat, recursive=True))
    seen, out = set(), []
    for p in sorted(paths):
        name = os.path.basename(p).lower()
        if p in seen or any(b in name for b in FONT_NAME_BLOCK):
            continue
        seen.add(p)
        out.append(p)
    return out


def digits_are_distinct(font: ImageFont.FreeTypeFont) -> list[np.ndarray] | None:
    """Render 1–9, normalize, and accept the font only if all glyphs render and are
    mutually distinct (rejects symbol fonts mapping every digit to the same box)."""
    canvases = []
    for d in "123456789":
        base = render_digit(font, d)
        if base.max() == 0:
            return None
        canvas = normalize_glyph(base)
        if canvas is None:
            return None
        canvases.append(canvas.astype(np.float32).ravel())
    for i in range(9):
        for j in range(i + 1, 9):
            a, b = canvases[i], canvases[j]
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
            if float(a @ b) / denom > 0.95:  # near-identical -> not real digits
                return None
    return canvases


def render_digit(font: ImageFont.FreeTypeFont, digit: str) -> np.ndarray:
    img = Image.new("L", (RENDER, RENDER), 0)  # black bg
    draw = ImageDraw.Draw(img)
    box = draw.textbbox((0, 0), digit, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    if tw <= 0 or th <= 0:
        return np.zeros((RENDER, RENDER), np.uint8)
    x = (RENDER - tw) / 2 - box[0]
    y = (RENDER - th) / 2 - box[1]
    draw.text((x, y), digit, fill=255, font=font)  # white ink
    return np.array(img)


def augment(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = mask.shape
    angle = rng.uniform(-8, 8)
    scale = rng.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    out = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    # Resolution degradation: real cell glyphs are only ~14–22 px tall, so a crisp
    # render must be downsampled (and re-upsampled) to reproduce the blobby aliasing.
    low = rng.randint(12, 24)
    out = cv2.resize(out, (low, low), interpolation=cv2.INTER_AREA)
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.4:
        noise = rng.uniform(8, 30)
        out = np.clip(out + np.random.normal(0, noise, out.shape), 0, 255).astype(np.uint8)
    # Stroke-weight jitter: real print ranges from hairline to heavy bold, which
    # shrinks the loops of 6/8/9 until they look solid. Cover that range.
    r = rng.random()
    if r < 0.45:
        k = rng.choice([2, 3, 3, 4])
        out = cv2.dilate(out, np.ones((k, k), np.uint8))
    elif r < 0.6:
        out = cv2.erode(out, np.ones((2, 2), np.uint8))
    return out


def build_dataset(aug_per_font: int = 16):
    fonts = find_fonts()
    if not fonts:
        raise RuntimeError("No usable system fonts found for training.")
    rng = random.Random(1234)
    np.random.seed(1234)
    X, y = [], []
    loaded = 0
    for path in fonts:
        try:
            font = ImageFont.truetype(path, RENDER - 8)
        except Exception:
            continue
        if digits_are_distinct(font) is None:
            continue  # skip symbol / non-Latin-digit fonts
        loaded += 1
        for d in "123456789":
            base = render_digit(font, d)
            for s in [base] + [augment(base, rng) for _ in range(aug_per_font)]:
                canvas = normalize_glyph(s)
                if canvas is None:
                    continue
                X.append(hog_features(canvas))
                y.append(int(d))
    print(f"Rendered from {loaded} usable fonts -> {len(X)} samples")
    return np.array(X), np.array(y)


def main() -> None:
    from sklearn.svm import SVC

    X, y = build_dataset()
    clf = SVC(kernel="rbf", C=10, gamma="scale")
    clf.fit(X, y)
    print(f"Train accuracy: {clf.score(X, y):.4f}")

    import joblib

    os.makedirs(os.path.dirname(os.path.abspath(MODEL_PATH)), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model -> {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
