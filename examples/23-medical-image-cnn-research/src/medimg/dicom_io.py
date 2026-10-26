"""Load DICOM images into normalized arrays, and strip direct identifiers.

Research/education code: not a medical device, not for clinical use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset

# Direct identifiers removed by deidentify(). This is NOT a complete
# de-identification profile (see DICOM PS3.15 Annex E); use a vetted tool
# for real data and follow your data-protection rules (e.g. GDPR, HIPAA).
IDENTIFYING_TAGS = [
    "PatientName", "PatientID", "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers",
    "OtherPatientIDs", "ReferringPhysicianName", "InstitutionName", "InstitutionAddress",
    "AccessionNumber", "StudyID", "OperatorsName", "PerformingPhysicianName",
]


def pixels(ds: Dataset) -> np.ndarray:
    """Pixel data as float32 in [0, 1]: rescale, window, fix MONOCHROME1 inversion."""
    img = ds.pixel_array.astype(np.float32)
    if img.ndim != 2:
        raise ValueError(f"expected a single 2D frame, got shape {img.shape}")
    img = img * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    center, width = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
    if center is not None and width is not None:
        center = float(center[0] if isinstance(center, pydicom.multival.MultiValue) else center)
        width = float(width[0] if isinstance(width, pydicom.multival.MultiValue) else width)
        lo, hi = center - width / 2, center + width / 2
    else:
        lo, hi = np.percentile(img, [0.5, 99.5])
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = 1.0 - img  # MONOCHROME1: high values are dark
    return img


def load(path: str | Path) -> np.ndarray:
    return pixels(pydicom.dcmread(path))


def deidentify(ds: Dataset) -> Dataset:
    for keyword in IDENTIFYING_TAGS:
        if keyword in ds:
            delattr(ds, keyword)
    ds.remove_private_tags()
    ds.PatientIdentityRemoved = "YES"
    return ds
