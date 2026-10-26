"""YOLO object detection with ONNX Runtime (no PyTorch needed at runtime).

Export a model once with Ultralytics (see README), e.g. `yolo11n.onnx`. Its
output has shape (1, 4 + num_classes, N): box center x/y, width, height in
input pixels, then one score per class. Only the classes we count are kept;
nothing about a person beyond "there is a person at this box" is computed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# COCO class ids used by the pretrained YOLO models.
COUNTED_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio, pad to size x size. Returns image, scale, (pad_x, pad_y)."""
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)  # gray padding, as in Ultralytics
    canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, scale, (left, top)


def postprocess(output: np.ndarray, scale: float, pad: tuple[float, float], frame_shape: tuple[int, int],
                conf: float = 0.35, iou: float = 0.5,
                classes: dict[int, str] = COUNTED_CLASSES) -> list[Detection]:
    preds = output[0].T  # (N, 4 + C)
    scores = preds[:, 4:]
    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]
    keep = (confidences >= conf) & np.isin(class_ids, list(classes))
    preds, class_ids, confidences = preds[keep], class_ids[keep], confidences[keep]
    if not len(preds):
        return []
    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x1 = (cx - bw / 2 - pad[0]) / scale
    y1 = (cy - bh / 2 - pad[1]) / scale
    boxes_xywh = np.stack([x1, y1, bw / scale, bh / scale], axis=1)
    # Class-aware NMS: offset boxes per class so different classes never suppress each other.
    offset = class_ids[:, None] * 10_000.0
    nms_boxes = boxes_xywh.copy()
    nms_boxes[:, :2] += offset
    picked = cv2.dnn.NMSBoxes(nms_boxes.tolist(), confidences.tolist(), conf, iou)
    h, w = frame_shape
    result = []
    for i in np.asarray(picked).reshape(-1):
        x, y, bw_, bh_ = boxes_xywh[i]
        box = (max(0.0, x), max(0.0, y), min(float(w), x + bw_), min(float(h), y + bh_))
        result.append(Detection(classes[int(class_ids[i])], float(confidences[i]), tuple(map(float, box))))
    return result


class OnnxYoloDetector:
    def __init__(self, model_path: str | None = None, size: int = 640, conf: float = 0.35, iou: float = 0.5,
                 session=None, classes: dict[int, str] = COUNTED_CLASSES) -> None:
        if session is None:
            import onnxruntime as ort

            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.size, self.conf, self.iou, self.classes = size, conf, iou, classes

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        image, scale, pad = letterbox(frame, self.size)
        blob = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        output = self.session.run(None, {self.input_name: blob})[0]
        return postprocess(output, scale, pad, frame.shape[:2], self.conf, self.iou, self.classes)
