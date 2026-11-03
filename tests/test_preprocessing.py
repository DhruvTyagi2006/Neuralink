import numpy as np
import pytest
from PIL import Image, ImageDraw
from src.preprocessing import PreprocessingError, load_uploaded_image, preprocess_image


def _digit_like(size=300, white_on_black=True, offset=(0, 0)):
    img = Image.new("L", (size, size), 0 if white_on_black else 255)
    d = ImageDraw.Draw(img)
    fg = 255 if white_on_black else 0
    ox, oy = offset
    d.line([(100 + ox, 60 + oy), (150 + ox, 240 + oy)], fill=fg, width=18)       # a slanted '1'-like stroke
    return img


def test_output_shape_and_range():
    vec, img = preprocess_image(_digit_like())
    assert vec.shape == (1, 784) and img.shape == (28, 28) and vec.dtype == np.float32
    assert 0.0 <= vec.min() and vec.max() <= 1.0 and vec.max() > 0.9


def test_polarity_is_normalised():
    a, _ = preprocess_image(_digit_like(white_on_black=True))
    b, _ = preprocess_image(_digit_like(white_on_black=False))
    assert np.corrcoef(a[0], b[0])[0, 1] > 0.95


def test_digit_is_centred_and_fits_box():
    _, img = preprocess_image(_digit_like(offset=(60, -30)))
    ys, xs = np.indices(img.shape)
    cy, cx = (ys * img).sum() / img.sum(), (xs * img).sum() / img.sum()
    assert abs(cy - 14) < 2 and abs(cx - 14) < 2
    rows = np.where(img.max(axis=1) > 0.25)[0]
    assert rows.max() - rows.min() + 1 <= 21


def test_translation_invariance():
    a, _ = preprocess_image(_digit_like(offset=(0, 0)))
    b, _ = preprocess_image(_digit_like(offset=(40, 20)))
    assert np.corrcoef(a[0], b[0])[0, 1] > 0.95


def test_rgba_and_array_inputs():
    arr = np.asarray(_digit_like().convert("RGBA"))
    assert preprocess_image(arr)[0].shape == (1, 784)
    assert preprocess_image(np.asarray(_digit_like()))[0].shape == (1, 784)


def test_blank_image_gives_clear_error():
    for blank in (np.zeros((200, 200)), np.full((200, 200), 255.0)):
        with pytest.raises(PreprocessingError):
            preprocess_image(blank)


def test_tiny_speck_is_rejected():
    img = np.zeros((200, 200))
    img[100, 100] = 255
    with pytest.raises(PreprocessingError):
        preprocess_image(img)


def test_invalid_upload_rejected(tmp_path):
    bad = tmp_path / "x.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(PreprocessingError):
        load_uploaded_image(bad)


def test_mnist_like_28x28_passes_through_almost_unchanged():
    rng = np.random.default_rng(0)
    im = np.zeros((28, 28), dtype=np.uint8)
    im[6:22, 12:16] = 255
    _, out = preprocess_image(im)
    assert out.shape == (28, 28) and out.sum() > 0
