"""Unit tests for image preprocessing."""
import io
import numpy as np
from PIL import Image
from api.inference.preprocess import (
    validate_image_bytes,
    quality_warnings,
    preprocess,
    five_crops,
    normalize,
)


def _make_img(w=256, h=256, color=(34, 139, 34)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def _img_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestValidateImage:
    def test_valid_jpeg(self):
        check = validate_image_bytes(_img_bytes(_make_img()))
        assert check.ok

    def test_valid_png(self):
        check = validate_image_bytes(_img_bytes(_make_img(), "PNG"))
        assert check.ok

    def test_corrupt_bytes(self):
        check = validate_image_bytes(b"this is not an image")
        assert not check.ok
        assert "Corrupt" in check.issue or "unsupported" in check.issue

    def test_too_small(self):
        check = validate_image_bytes(_img_bytes(_make_img(w=32, h=32)))
        assert not check.ok
        assert "small" in check.issue.lower()

    def test_empty_bytes(self):
        check = validate_image_bytes(b"")
        assert not check.ok


class TestQualityWarnings:
    def test_normal_image(self):
        """A solid single-color image may trigger low_variance. Use varied pixels."""
        import random
        random.seed(42)
        img = Image.new("RGB", (256, 256))
        pixels = [(random.randint(80, 200), random.randint(80, 200), random.randint(80, 200)) for _ in range(256*256)]
        img.putdata(pixels)
        assert quality_warnings(img) == []

    def test_very_dark(self):
        img = _make_img(color=(5, 5, 5))
        assert "very_dark" in quality_warnings(img)

    def test_overexposed(self):
        img = _make_img(color=(250, 250, 250))
        assert "overexposed" in quality_warnings(img)


class TestPreprocess:
    def test_output_shape(self):
        img = _make_img()
        out = preprocess(img, size=256)
        assert out.shape == (1, 3, 256, 256)
        assert out.dtype == np.float32

    def test_different_size(self):
        img = _make_img(w=512, h=512)
        out = preprocess(img, size=224)
        assert out.shape == (1, 3, 224, 224)


class TestTTACrops:
    def test_five_crops(self):
        img = _make_img(w=512, h=512)
        crops = five_crops(img, n=5)
        assert len(crops) == 5
        for c in crops:
            assert isinstance(c, Image.Image)

    def test_fewer_crops(self):
        img = _make_img()
        crops = five_crops(img, n=3)
        assert len(crops) == 3


class TestNormalize:
    def test_zero_input(self):
        arr = np.zeros((3,), dtype=np.float32)
        result = normalize(arr)
        assert result.shape == (3,)
        # Should be -mean/std
        assert result[0] < 0
