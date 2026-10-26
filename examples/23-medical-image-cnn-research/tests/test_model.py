import numpy as np

from medimg import data
from medimg.evaluate import binary_metrics, grad_cam
from medimg.model import resnet50_transfer, small_cnn
from medimg.train import run


def test_binary_metrics():
    m = binary_metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.1, 0.7]))
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5 and m["accuracy"] == 0.5


def test_small_cnn_learns_synthetic_blobs(tmp_path):
    x, y = data.synthetic(160, size=48, seed=1)
    report = run(x, y, "small", epochs=30, seed=1, out=str(tmp_path / "m.keras"))
    assert report["test_images"] == 32
    assert report["accuracy"] >= 0.85, report
    assert (tmp_path / "m.keras").exists()


def test_grad_cam_shape_and_range():
    model = small_cnn(48)
    x, _ = data.synthetic(2, size=48)
    cam = grad_cam(model, x[1, ..., 0])
    assert cam.shape == (48, 48)
    assert 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6


def test_resnet50_transfer_builds_on_cpu_without_download():
    model = resnet50_transfer(64, weights=None)
    out = model.predict(np.zeros((2, 64, 64, 1), np.float32), verbose=0)
    assert out.shape == (2, 1)
    assert not model.get_layer("resnet50").trainable
