import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from medimg import data, dicom_io


def make_dicom(path, pixels, photometric="MONOCHROME2", window=None, slope=1.0, intercept=0.0):
    """Write a small synthetic DICOM file (no real patient data)."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.PatientName, ds.PatientID, ds.InstitutionName = "Test^Synthetic", "SYN-001", "Example Hospital"
    ds.Modality = "OT"
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, photometric
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
    if window:
        ds.WindowCenter, ds.WindowWidth = window
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.add_new(0x00090010, "LO", "PRIVATE CREATOR")  # a private tag
    ds.save_as(path, enforce_file_format=True)
    return path


def test_load_applies_window_and_normalizes(tmp_path):
    px = np.tile(np.linspace(0, 4000, 32), (32, 1))
    img = dicom_io.load(make_dicom(tmp_path / "a.dcm", px, window=(2000, 2000)))
    assert img.shape == (32, 32) and img.dtype == np.float32
    assert img[0, 0] == 0.0 and img[0, -1] == 1.0
    assert img[0, 16] == pytest.approx(0.52, abs=0.03)


def test_rescale_and_monochrome1_inversion(tmp_path):
    px = np.tile(np.linspace(0, 1000, 16), (16, 1))
    mono2 = dicom_io.load(make_dicom(tmp_path / "b.dcm", px, slope=2.0, intercept=-1000.0))
    mono1 = dicom_io.load(make_dicom(tmp_path / "c.dcm", px, photometric="MONOCHROME1",
                                     slope=2.0, intercept=-1000.0))
    assert mono2[0, -1] > mono2[0, 0]
    np.testing.assert_allclose(mono1, 1 - mono2, atol=1e-6)


def test_deidentify_removes_direct_identifiers(tmp_path):
    import pydicom

    ds = pydicom.dcmread(make_dicom(tmp_path / "d.dcm", np.zeros((8, 8))))
    dicom_io.deidentify(ds)
    assert "PatientName" not in ds and "PatientID" not in ds and "InstitutionName" not in ds
    assert ds.PatientIdentityRemoved == "YES"
    assert not any(elem.tag.is_private for elem in ds)


def test_load_folder_builds_labeled_tensor(tmp_path):
    x, _ = data.synthetic(4, size=48)
    for i, name in enumerate(["normal", "normal", "finding", "finding"]):
        (tmp_path / name).mkdir(exist_ok=True)
        make_dicom(tmp_path / name / f"{i}.dcm", x[i, ..., 0] * 4095)
    images, labels = data.load_folder(tmp_path, size=32)
    assert images.shape == (4, 32, 32, 1)
    assert labels.tolist() == [0, 0, 1, 1]
    with pytest.raises(ValueError):
        data.load_folder(tmp_path / "missing", size=32)
