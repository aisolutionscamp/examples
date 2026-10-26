# Visual defect detection: OpenCV patch features + IsolationForest, exported to ONNX

Use case: [Autonomous Quality Control in Manufacturing](https://aisolutionscamp.io/projects/29)

Anomaly detection for visual inspection: train on images of **good** parts only, flag images
that look different, and show where. A classical, explainable baseline that runs on a CPU.

## How it works

```
good part images
  -> preprocess (gray, 256 x 256, median brightness = 0.5)
  -> 225 overlapping 32 x 32 patches (stride 16), 8 features each:
     mean, std, 2nd/98th percentile, Laplacian variance, mean/max gradient, Canny edge density
  -> StandardScaler + IsolationForest (fit on good patches)
  -> calibrate threshold on held-out images (max false-reject rate, default 1 %)
  -> export scaler + forest to ONNX (skl2onnx)

new image -> same features -> ONNX Runtime -> patch scores
  -> image score = max patch score, defect = score >= threshold, box = most unusual patch
```

- `src/qc/features.py` preprocessing and patch features (OpenCV)
- `src/qc/detector.py` training, calibration, scoring, heatmap
- `src/qc/onnx_export.py` ONNX export and an onnxruntime scorer
- `src/qc/synthetic.py` synthetic surfaces with scratches, spots and cracks
- `src/qc/mvtec.py` file lists for the MVTec AD folder layout
- `src/qc/cli.py` command line

## Setup and run

```bash
pip install -r requirements.txt
cd src
python -m qc.cli demo                      # whole flow on synthetic images

python -m qc.cli train --good data/train/good --val-good data/val/good \
    --val-defect data/val/defect --max-false-reject 0.01 \
    --out model.joblib --onnx model.onnx   # prints the calibrated threshold
python -m qc.cli score --model model.joblib part1.png part2.png
```

Without `--val-defect` the threshold is the 99th percentile (for 1 %) of the good-part scores.
With defects it is the threshold with the highest defect recall whose false-reject rate stays
within the limit. Calibration images must not be the training images.

To use the ONNX model, keep the threshold from training and run
`OnnxScorer("model.onnx", threshold).predict(image)`; only `opencv-python-headless`, `numpy`
and `onnxruntime` are needed at inference time.

### Public benchmark: MVTec AD

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) has 15 object and texture
categories with good training images and labeled test defects. `qc.mvtec.mvtec_split(root,
"bottle")` returns the file lists. **License: CC BY-NC-SA 4.0, non-commercial use only.**
This example has not been benchmarked on MVTec AD; classical features like these do well on
some textures and poorly on objects with complex shapes.

## Tests

```bash
pytest     # or: scripts/test-example.sh examples/29-visual-anomaly-detection
```

They train on synthetic images and check that defects score higher than good parts (ROC AUC >
0.95), that the most unusual patch lies on the defect for each defect type, that calibration
keeps the false-reject limit, that ONNX scores match scikit-learn, the MVTec file layout, and
the CLI.

## Limitations

- Hand-crafted features see contrast and texture changes. Subtle defects (slight dents, colour
  shifts, missing small components) need deep features: PatchCore-style methods with a
  pretrained CNN backbone, e.g. via the anomalib library.
- Works only for aligned images under constant lighting. Position changes look like defects.
- Patch size limits the smallest detectable defect. Tile high-resolution images instead of
  downscaling them.
- The synthetic test images are much easier than real parts.

## For production

- Fix the camera, lens and lighting first; this matters more than the model.
- Collect a few hundred good images across shifts and material batches, and every real defect you can find.
- Measure false-reject rate and recall per defect type on a separate labeled test set.
- Log scores and operator overrides; recalibrate when material, lighting or the camera changes.
- Keep a human decision for borderline scores until the measured rates are accepted by quality management.
