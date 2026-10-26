import numpy as np

from fftdeep.frequency.spectrum import log_spectrum


def test_constant_image_has_empty_spectrum():
    imgs = np.full((2, 32, 32, 3), 100, dtype=np.uint8)
    assert np.allclose(log_spectrum(imgs), 0, atol=1e-6)


def test_cosine_peak_is_at_expected_frequency():
    n = np.arange(32)
    row = 127.5 + 100 * np.cos(2 * np.pi * 4 * n / 32)          # 4 cycles across the width
    img = np.tile(row, (32, 1))[None, :, :, None].repeat(3, axis=3).astype(np.uint8)
    spec = log_spectrum(img)
    r, c = np.unravel_index(spec[0].argmax(), spec[0].shape)
    assert spec.shape == (1, 32, 32)
    assert r == 16 and c in (12, 20)                           # centre row, 4 bins either side of centre