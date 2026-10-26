# CNN image classification on DICOM data (research example)

> **Research and education example. Not a medical device. Not for diagnosis, triage or any clinical decision.**
> The code has not been validated on patient data, and nothing in this folder makes a claim about diagnostic
> performance.

It shows the technical pieces of a medical image classification experiment with TensorFlow/Keras and pydicom:

- loading DICOM pixel data correctly (rescale slope/intercept, windowing, MONOCHROME1 inversion),
- removing direct identifiers before data enters an experiment,
- a small CNN (CPU, seconds) and a ResNet50 transfer-learning model,
- evaluation with sensitivity and specificity, and Grad-CAM heatmaps.

The tests use synthetic DICOM files and synthetic images where the "finding" is a bright blob.

## Architecture

```
DICOM files (normal/, finding/) ──► deidentify() ──► pixels(): rescale, window, invert, [0, 1]
   ──► resize ──► augmentation (rotation, zoom, contrast; no flip)
   ──► small_cnn  or  ResNet50 (ImageNet weights, frozen) + new head ──► sigmoid probability
   ──► sensitivity / specificity on a held-out split, Grad-CAM heatmap per image
```

- `dicom_io.py`: `load`, `pixels`, `deidentify`.
- `data.py`: folder loader and the synthetic image generator.
- `model.py`: `small_cnn`, `resnet50_transfer`.
- `evaluate.py`: `binary_metrics`, `grad_cam`.
- `train.py`: split, train with early stopping, report.

## Setup and run

```bash
pip install -r requirements.txt          # tensorflow-cpu; use "tensorflow" instead for a GPU machine
cd src
python -m medimg.train --synthetic --size 64 --epochs 15             # synthetic smoke run
python -m medimg.train --data /path/to/data --model resnet50 --size 224 --epochs 20 --out model.keras
```

`--data` expects `normal/*.dcm` and `finding/*.dcm`. The ResNet50 model downloads ImageNet weights from the
Keras servers on first use (about 95 MB). The output is JSON with `sensitivity`, `specificity` and `accuracy`
at threshold 0.5.

Tests (offline, CPU only, no weight download): `pytest`.

## Data

Use only data you are allowed to use, with the approvals your institution requires. Public research datasets
include the RSNA Pneumonia Detection Challenge (DICOM) and NIH ChestX-ray14 (PNG); each has its own terms of use.
Do not commit images to this repository.

## Limitations

- `deidentify()` removes a short list of direct identifiers and private tags. It is not a complete
  de-identification (DICOM PS3.15 Annex E), and burned-in text in the pixels is not touched.
- `train.split` is random by image. With real data, split by patient, or images of the same patient end up in
  training and test and the metrics are too optimistic.
- Binary labels only; no localization, no multi-label findings, single frame per file.
- Results on synthetic blobs say nothing about medical images.

## Before any further use

- Patient-level splits, an external test set from another site, confidence intervals for sensitivity and
  specificity, and a review of Grad-CAM maps for shortcuts (markers, text, device type).
- Infrastructure that meets your data-protection requirements (for example GDPR, HIPAA).
- Any use in patient care requires regulatory clearance as a medical device (for example EU MDR, FDA). This
  example is not designed or documented for that.
