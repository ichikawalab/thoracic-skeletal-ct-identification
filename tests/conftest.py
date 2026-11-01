"""Test helpers.

Load the numbered stage scripts by file path, matching their CLI entry points.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_stage_script(relative_path: str) -> types.ModuleType:
    path = os.path.join(REPO_ROOT, relative_path)
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_ct_series(directory, n=9, varying_rescale=False, orientation=(1, 0, 0, 0, 1, 0)):
    directory.mkdir(parents=True, exist_ok=True)
    study_uid, series_uid = generate_uid(), generate_uid()
    frame_uid = generate_uid()
    normal = np.cross(orientation[:3], orientation[3:])
    headers = []
    for k in range(n):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        path = directory / f"{k:04}.dcm"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.Modality, ds.Manufacturer, ds.PatientID = "CT", "SIEMENS", "1"
        ds.FrameOfReferenceUID = frame_uid
        ds.ClinicalTrialTimePointID, ds.AcquisitionNumber = "T0", 1
        ds.ManufacturerModelName, ds.ConvolutionKernel = "synthetic", "B30f"
        ds.GantryDetectorTilt, ds.PatientPosition = 0, "HFS"
        ds.StudyDate, ds.StudyTime, ds.SeriesNumber, ds.InstanceNumber = "20000101", "120000", 1, k + 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.ImageOrientationPatient = list(orientation)
        ds.ImagePositionPatient = (np.array([-100., -120., -250.]) + k * 2.5 * normal).tolist()
        ds.PixelSpacing, ds.SliceThickness = [0.7, 0.8], 2.5
        ds.Rows, ds.Columns = 6, 8
        ds.PhotometricInterpretation, ds.SamplesPerPixel = "MONOCHROME2", 1
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.RescaleSlope = 1 + k * 0.125 if varying_rescale else 1
        ds.RescaleIntercept = -1024 + (k % 3) * 0.25 if varying_rescale else -1024
        ds.RescaleType = "HU"
        pixels = np.arange(48, dtype=np.uint16).reshape(6, 8) + 1000 + k * 50
        ds.PixelData = pixels.tobytes()
        ds.save_as(path, write_like_original=False)
        headers.append(ds)
    return headers
