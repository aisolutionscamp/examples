import cv2
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from qc.cli import main as cli_main
from qc.detector import PatchAnomalyDetector
from qc.features import PATCH, patch_features, patch_grid, preprocess
from qc.mvtec import mvtec_split
from qc.onnx_export import OnnxScorer, export_onnx
from qc.synthetic import add_defect, good_image


@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(0)
    det = PatchAnomalyDetector(n_estimators=100).fit([good_image(rng) for _ in range(12)])
    return det, rng


def test_features_shape_and_preprocess_handles_color():
    color = cv2.cvtColor(good_image(np.random.default_rng(1), size=300), cv2.COLOR_GRAY2BGR)
    gray = preprocess(color)
    assert gray.shape == (256, 256) and gray.dtype == np.float32 and gray.max() <= 1
    assert patch_features(gray).shape == (len(patch_grid()), 8)


def test_defects_score_higher_than_good_parts(trained):
    # Synthetic sanity check of the pipeline, not a claim about real parts.
    det, _ = trained
    rng = np.random.default_rng(42)
    good = [det.score(good_image(rng)) for _ in range(15)]
    bad = [det.score(add_defect(good_image(rng), rng)[0]) for _ in range(15)]
    assert roc_auc_score([0] * 15 + [1] * 15, good + bad) > 0.95


@pytest.mark.parametrize("kind", ["scratch", "spot", "crack"])
def test_most_unusual_patch_is_at_the_defect(trained, kind):
    det, _ = trained
    rng = np.random.default_rng(7)
    img, (cx, cy) = add_defect(good_image(rng), rng, kind)
    y, x = patch_grid()[int(det.patch_scores(img).argmax())]
    assert x - 8 <= cx <= x + PATCH + 8 and y - 8 <= cy <= y + PATCH + 8
    assert det.heatmap(img).shape == (15, 15)


def test_calibration_respects_false_reject_limit(trained):
    det, _ = trained
    rng = np.random.default_rng(3)
    val_good = [good_image(rng) for _ in range(20)]
    val_bad = [add_defect(good_image(rng), rng)[0] for _ in range(10)]
    th = det.calibrate(val_good, val_bad, max_false_reject=0.05)
    assert th.false_reject_rate <= 0.05
    assert th.defect_recall >= 0.8
    unsupervised = det.calibrate(val_good, max_false_reject=0.05)
    assert unsupervised.defect_recall is None and unsupervised.false_reject_rate <= 0.1
    result = det.predict(val_bad[0])
    assert set(result) >= {"score", "threshold", "defect", "box"}


def test_predict_requires_calibration():
    with pytest.raises(RuntimeError):
        PatchAnomalyDetector().predict(np.zeros((64, 64), np.uint8))


def test_onnx_matches_scikit_learn(trained, tmp_path):
    det, _ = trained
    rng = np.random.default_rng(5)
    det.calibrate([good_image(rng) for _ in range(5)])
    path = str(tmp_path / "m.onnx")
    export_onnx(det, path)
    scorer = OnnxScorer(path, det.threshold.value)
    img = add_defect(good_image(rng), rng)[0]
    assert np.allclose(scorer.patch_scores(img), det.patch_scores(img), atol=1e-4)
    assert scorer.predict(img)["defect"] == det.predict(img)["defect"]


def test_mvtec_layout(tmp_path):
    for sub in ["bottle/train/good", "bottle/test/good", "bottle/test/broken_large", "bottle/test/contamination"]:
        (tmp_path / sub).mkdir(parents=True)
        cv2.imwrite(str(tmp_path / sub / "000.png"), np.zeros((8, 8), np.uint8))
    split = mvtec_split(tmp_path, "bottle")
    assert [len(split[k]) for k in ("train_good", "test_good", "test_defect")] == [1, 1, 2]


def test_cli_train_and_score(tmp_path, capsys):
    rng = np.random.default_rng(9)
    for name, n, defect in [("train", 8, False), ("val_good", 6, False), ("val_bad", 4, True)]:
        (tmp_path / name).mkdir()
        for i in range(n):
            img = add_defect(good_image(rng), rng)[0] if defect else good_image(rng)
            cv2.imwrite(str(tmp_path / name / f"{i}.png"), img)
    model = str(tmp_path / "m.joblib")
    cli_main(["train", "--good", str(tmp_path / "train"), "--val-good", str(tmp_path / "val_good"),
              "--val-defect", str(tmp_path / "val_bad"), "--max-false-reject", "0.2",
              "--out", model, "--onnx", str(tmp_path / "m.onnx")])
    cli_main(["score", "--model", model, str(tmp_path / "val_bad" / "0.png")])
    out = capsys.readouterr().out
    assert '"defect": true' in out and (tmp_path / "m.onnx").exists()
