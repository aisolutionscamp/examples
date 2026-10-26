import numpy as np
import pytest

from counting.detector import OnnxYoloDetector, letterbox, postprocess


def yolo_output(rows, num_classes=80):
    """Build a (1, 4 + C, N) array like a YOLOv8/YOLO11 ONNX export. rows: (cx, cy, w, h, class, score)."""
    out = np.zeros((1, 4 + num_classes, len(rows)), np.float32)
    for n, (cx, cy, w, h, cls, score) in enumerate(rows):
        out[0, :4, n] = (cx, cy, w, h)
        out[0, 4 + cls, n] = score
    return out


def test_letterbox_scale_and_padding():
    img, scale, pad = letterbox(np.zeros((360, 640, 3), np.uint8), 640)
    assert img.shape == (640, 640, 3) and scale == 1.0 and pad == (0, 140)
    assert img[0, 0, 0] == 114 and img[300, 300, 0] == 0


def test_postprocess_filters_classes_scores_and_overlaps():
    out = yolo_output([
        (100, 100, 40, 80, 0, 0.9),   # person
        (102, 101, 40, 80, 0, 0.6),   # duplicate of the same person -> removed by NMS
        (102, 101, 40, 80, 2, 0.7),   # car at the same place -> kept (other class)
        (300, 300, 50, 50, 0, 0.2),   # low confidence -> dropped
        (400, 300, 50, 50, 15, 0.95),  # cat: not a counted class -> dropped
    ])
    dets = postprocess(out, scale=1.0, pad=(0, 0), frame_shape=(640, 640))
    assert sorted(d.label for d in dets) == ["car", "person"]
    person = next(d for d in dets if d.label == "person")
    assert person.box == pytest.approx((80, 60, 120, 140))


def test_postprocess_maps_back_to_frame_coordinates():
    out = yolo_output([(320, 320, 64, 64, 0, 0.8)])
    det = postprocess(out, scale=0.5, pad=(0, 140), frame_shape=(720, 1280))[0]
    assert det.box == pytest.approx((576, 296, 704, 424))


class FakeInput:
    name = "images"


class FakeSession:
    """Stands in for onnxruntime.InferenceSession: no model file, no download."""

    def __init__(self, output):
        self.output, self.calls = output, []

    def get_inputs(self):
        return [FakeInput()]

    def run(self, _names, feeds):
        self.calls.append(feeds["images"])
        return [self.output]


def test_detector_feeds_normalized_nchw_tensor():
    session = FakeSession(yolo_output([(320, 320, 64, 128, 0, 0.9)]))
    dets = OnnxYoloDetector(session=session)(np.full((480, 640, 3), 255, np.uint8))
    blob = session.calls[0]
    assert blob.shape == (1, 3, 640, 640) and blob.dtype == np.float32
    assert blob.max() == pytest.approx(1.0)
    assert len(dets) == 1 and dets[0].label == "person"
