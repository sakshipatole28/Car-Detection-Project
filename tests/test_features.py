import cv2
import numpy as np
import pytest

from cardet.features import decode_image, to_gray


def test_window_features_match_patch_features(extractor, rng):
    """Train/serve parity: a window cut from a big image == the same crop as a patch."""
    img = rng.integers(0, 256, (97, 233), dtype=np.uint8)
    feats, tl = extractor.windows(img, stride=7)
    assert feats.shape[1] == extractor.n_features == 972
    for k in rng.choice(len(feats), 15, replace=False):
        x, y = tl[k]
        np.testing.assert_array_equal(feats[k], extractor.patch(img[y : y + 40, x : x + 100]))


def test_window_grid_covers_image(extractor):
    img = np.zeros((60, 140), np.uint8)
    feats, tl = extractor.windows(img, stride=10)
    # x in {0..40}, y in {0..20}
    assert len(feats) == 5 * 3
    assert tl[:, 0].max() == 40 and tl[:, 1].max() == 20
    assert tl[1].tolist() == [10, 0]  # x varies fastest


def test_image_smaller_than_window_returns_nothing(extractor):
    feats, tl = extractor.windows(np.zeros((30, 200), np.uint8), stride=4)
    assert feats.shape == (0, 972) and tl.shape == (0, 2)


def test_patch_resizes_any_input(extractor, rng):
    assert extractor.patch(rng.integers(0, 256, (80, 200), dtype=np.uint8)).shape == (972,)


def test_extractor_is_picklable(extractor):
    import pickle

    extractor.patch(np.zeros((40, 100), np.uint8))  # instantiate the cv2 object
    clone = pickle.loads(pickle.dumps(extractor))
    assert clone.n_features == extractor.n_features


@pytest.mark.parametrize(
    "img",
    [
        np.full((10, 10, 3), 128, np.uint8),
        np.full((10, 10, 4), 128, np.uint8),
        np.full((10, 10), 40000, np.uint16),
    ],
)
def test_to_gray_normalises_formats(img):
    g = to_gray(img)
    assert g.dtype == np.uint8 and g.ndim == 2


def test_decode_png_and_reject_garbage():
    ok, buf = cv2.imencode(".png", np.zeros((40, 100, 3), np.uint8))
    assert decode_image(buf.tobytes()).shape == (40, 100)
    with pytest.raises(ValueError):
        decode_image(b"definitely not an image")
