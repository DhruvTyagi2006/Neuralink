"""Turn a user's drawing / uploaded photo into an MNIST-style 784-feature vector.

MNIST digits are: white strokes on a black background, scaled to fit a 20x20 box
(aspect ratio preserved) and placed in a 28x28 image so that the centre of mass of
the pixels is at the image centre.  This module reproduces that recipe so that
user input matches what the network was trained on.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .config import IMAGE_SIDE

BOX = 20                # the digit is fitted into a 20x20 box inside the 28x28 frame
INK_THRESHOLD = 0.25    # relative intensity above which a pixel counts as "ink"
MIN_INK_PIXELS = 12
MIN_CONTRAST = 30       # minimum (max - median) grey-level span, out of 255


class PreprocessingError(ValueError):
    """Raised when the input has no recognisable digit (blank, too faint, wrong type)."""


def to_grayscale(img) -> np.ndarray:
    """PIL image / ndarray (H,W), (H,W,3), (H,W,4)  ->  float array (H,W) with values 0..255.
    Transparent pixels are composited on white (typical for exported PNG drawings)."""
    if isinstance(img, Image.Image):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return np.asarray(Image.alpha_composite(bg, rgba).convert("L"), dtype=np.float64)
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        if arr.shape[2] == 4:
            alpha = arr[..., 3:4].astype(np.float64) / 255.0
            arr = arr[..., :3] * alpha + 255.0 * (1 - alpha)
        rgb = arr[..., :3].astype(np.float64)
        return rgb @ np.array([0.299, 0.587, 0.114])
    raise PreprocessingError(f"Unsupported image array shape {arr.shape}")


def normalise_polarity(gray: np.ndarray) -> np.ndarray:
    """Make the digit bright on a dark background. The background is the *median* pixel
    (ink covers a minority of the image); if it is bright we invert."""
    if np.median(gray) > 127:
        gray = 255.0 - gray
    return gray


def _isolate_digit(gray: np.ndarray) -> np.ndarray:
    bg = np.median(gray)
    peak = gray.max()
    if peak - bg < MIN_CONTRAST:
        raise PreprocessingError("No visible digit found - the image looks blank or too faint.")
    return np.clip((gray - bg) / (peak - bg), 0.0, 1.0)  # background -> 0, strongest ink -> 1


def _bounding_box(g: np.ndarray):
    mask = g > INK_THRESHOLD
    if mask.sum() < MIN_INK_PIXELS:
        raise PreprocessingError("The drawing is too small or too faint to recognise a digit.")
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def _center_of_mass(a: np.ndarray):
    total = a.sum()
    ys, xs = np.indices(a.shape)
    return (ys * a).sum() / total, (xs * a).sum() / total


def preprocess_image(img):
    """Full pipeline. Returns (vector, image28):
        vector  : float32 array of shape (1, 784), values in [0, 1]
        image28 : float32 array of shape (28, 28) (what the network actually sees)
    """
    g = _isolate_digit(normalise_polarity(to_grayscale(img)))
    r0, r1, c0, c1 = _bounding_box(g)
    crop = g[r0:r1, c0:c1]                                    # remove empty borders
    h, w = crop.shape
    scale = BOX / max(h, w)                                   # preserve the aspect ratio
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = np.asarray(
        Image.fromarray(crop.astype(np.float32)).resize((new_w, new_h), Image.Resampling.LANCZOS),
        dtype=np.float32)
    resized = np.clip(resized, 0.0, 1.0)
    if resized.max() > 0:
        resized /= resized.max()

    cy, cx = _center_of_mass(resized)                         # centre by centre of mass (MNIST convention)
    top = int(np.clip(round(IMAGE_SIDE / 2 - cy), 0, IMAGE_SIDE - new_h))
    left = int(np.clip(round(IMAGE_SIDE / 2 - cx), 0, IMAGE_SIDE - new_w))
    out = np.zeros((IMAGE_SIDE, IMAGE_SIDE), dtype=np.float32)
    out[top:top + new_h, left:left + new_w] = resized
    return out.reshape(1, -1), out


def load_uploaded_image(file) -> Image.Image:
    """Validate an uploaded file object and return a PIL image."""
    try:
        img = Image.open(file)
        img.load()
    except Exception as exc:
        raise PreprocessingError("The file is not a readable image (use PNG, JPG, BMP or WEBP).") from exc
    if min(img.size) < 8:
        raise PreprocessingError("The image is too small - it must be at least 8x8 pixels.")
    return img
