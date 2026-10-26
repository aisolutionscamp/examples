import cv2
import numpy as np

from bgremove.__main__ import process_folder
from bgremove.remover import Settings, foreground_mask, on_white, remove_background


def product_photo(h=240, w=320, seed=0):
    """Light, slightly noisy studio background with a soft shadow gradient and a 'product'."""
    rng = np.random.default_rng(seed)
    gradient = np.linspace(235, 250, w, dtype=np.float32)[None, :, None]
    img = np.clip(np.broadcast_to(gradient, (h, w, 3)) + rng.normal(0, 2, (h, w, 3)), 0, 255).astype(np.uint8)
    truth = np.zeros((h, w), np.uint8)
    cv2.rectangle(truth, (110, 60), (210, 190), 255, -1)  # bottle body
    cv2.circle(truth, (160, 55), 25, 255, -1)  # cap
    body = np.zeros_like(img)
    body[:] = (40, 60, 200)  # red-ish in BGR
    body[100:130, 125:195] = (232, 232, 232)  # light label, close to the background color
    img = np.where(truth[:, :, None] == 255, body, img)
    return img, truth


def iou(a, b):
    a, b = a > 127, b > 127
    return (a & b).sum() / (a | b).sum()


def test_mask_matches_synthetic_product():
    img, truth = product_photo()
    mask = foreground_mask(img)
    assert iou(mask, truth) > 0.9


def test_rgba_corners_transparent_product_opaque():
    img, _ = product_photo()
    rgba = remove_background(img)
    assert rgba.shape == (240, 320, 4)
    assert rgba[0, 0, 3] == 0 and rgba[-1, -1, 3] == 0
    assert rgba[150, 160, 3] == 255


def test_light_label_inside_product_is_kept():
    img, _ = product_photo()
    assert foreground_mask(img)[115, 160] == 255
    assert foreground_mask(img, Settings(fill_holes=False))[115, 160] == 0


def test_plain_image_gives_empty_mask():
    img = np.full((100, 100, 3), 240, np.uint8)
    assert foreground_mask(img).max() == 0


def test_downscaled_path_keeps_full_resolution():
    img, truth = product_photo()
    big = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_NEAREST)
    big_truth = cv2.resize(truth, (1280, 960), interpolation=cv2.INTER_NEAREST)
    mask = foreground_mask(big, Settings(max_side=400))
    assert mask.shape == (960, 1280)
    assert iou(mask, big_truth) > 0.85


def test_keep_largest_drops_small_second_object():
    img, truth = product_photo()
    cv2.circle(img, (290, 210), 8, (0, 0, 0), -1)
    mask = foreground_mask(img, Settings(keep_largest=True))
    assert mask[210, 290] == 0
    assert iou(mask, truth) > 0.9


def test_on_white_background_is_white():
    img, _ = product_photo()
    out = on_white(img, foreground_mask(img))
    assert (out[0, 0] == 255).all()


def test_batch_cli(tmp_path):
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    for i in range(2):
        cv2.imwrite(str(src / f"p{i}.jpg"), product_photo(seed=i)[0])
    (src / "notes.txt").write_text("ignored")
    written = process_folder(src, dst, Settings())
    assert [p.name for p in written] == ["p0.png", "p1.png"]
    assert cv2.imread(str(written[0]), cv2.IMREAD_UNCHANGED).shape[2] == 4
    white = process_folder(src, tmp_path / "white", Settings(), white=True)
    assert white[0].suffix == ".jpg"
