"""DICOM series handling for the inventory and conversion stages: slice geometry along the normal, duplicate
positions, pixel decoding from the low BitsStored bits, all-slice validation of conventional CT series, and the
private SQLite inventory index used for the global UID audit.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom

from common.provenance import TIME_POINTS, file_hash

DUPLICATE_TOL_MM = 0.01  # candidate duplicate positions; matching geometry and HU are checked separately
POSITION_TOL_MM = 0.05
HU_TOL = 0.01
ROUNDING_MM = 1e-9  # Absorb floating-point roundoff at distance thresholds.
CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
TRANSFER_SYNTAXES = {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1"}

# Representative values are stored with the list of tags that varied across slices.
SUMMARY_TAGS = [
    "PatientID", "ClinicalTrialTimePointID", "StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID",
    "StudyDate", "SeriesNumber", "SeriesDescription", "ClinicalTrialSiteID",
    "Modality", "SOPClassUID", "ImageType", "Manufacturer", "ManufacturerModelName", "SoftwareVersions", "StationName",
    "ConvolutionKernel", "FilterType", "PatientPosition", "GantryDetectorTilt", "SliceThickness",
    "PixelSpacing", "Rows", "Columns", "ImageOrientationPatient", "ReconstructionDiameter",
    "KVP", "XRayTubeCurrent", "Exposure", "ExposureTime", "AcquisitionNumber",
    "SamplesPerPixel", "PhotometricInterpretation", "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation",
    "RescaleSlope", "RescaleIntercept", "RescaleType",
]
HEADER_TAGS = SUMMARY_TAGS + ["SOPInstanceUID", "ImagePositionPatient", "InstanceNumber", "NumberOfFrames",
                              "MultienergyCTAcquisition", "ModalityLUTSequence"]
READ_TAGS = [pydicom.tag.Tag(tag) for tag in HEADER_TAGS]
AUDIT_FIELDS = [
    "validation_status", "validation_reason", "observed_patient_ids", "observed_timepoints",
    "n_unread", "varied_tags", "n_unique_slices", "n_duplicates",
    "spacing_median_mm", "spacing_max_mm", "spacing_max_dev_mm", "extent_mm", "position_first_lps",
    "position_last_lps", "inplane_max_dev_mm", "grid_max_dev_mm", "normal_z",
    "flag_instance_mismatch", "flag_rescale_varies", "flag_acquisition_varies", "source_sha256",
]
CONSTANT_GEOMETRY = {"ImageOrientationPatient": 6, "PixelSpacing": 2, "Rows": 1, "Columns": 1, "SliceThickness": 1}
# Stored bits and signedness may differ between slices: every slice's samples are taken from the low
# BitsStored bits of its own header, and the conversion verifies every slice against the source.
CONSTANT_ENCODING = ["SamplesPerPixel", "PhotometricInterpretation", "BitsAllocated"]
CONSTANT_CONTEXT = ["StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID", "Manufacturer",
                    "ManufacturerModelName", "ConvolutionKernel", "FilterType", "PatientPosition"]
INVENTORY_FIELDS = (["pid", "n_series_subject", "study_folder", "series_folder", "n_files"]
                    + SUMMARY_TAGS + ["TransferSyntaxUID"]
                    + [f for f in AUDIT_FIELDS if f != "validation_status"]
                    + ["local_validation_status", "validation_status"])


class InvalidSeries(ValueError):
    """A known invalid or unsupported input condition, distinct from an unexpected program failure."""


# ---- Slice geometry ----

@dataclass
class SpacingStats:
    n_input: int
    n_unique: int               # representative positions, before checking duplicate content
    n_duplicates: int
    spacing_median_mm: float
    spacing_max_dev_mm: float   # max |spacing_i - median|
    extent_mm: float
    order: np.ndarray           # indices of unique slices sorted along the normal
    inplane_max_dev_mm: float = 0.0


def unit_normal(orientation: np.ndarray) -> np.ndarray:
    """Unit slice normal of an ImageOrientationPatient; the stored vectors are only orthonormal within 1e-4."""
    normal = np.cross(orientation[:3], orientation[3:])
    return normal / np.linalg.norm(normal)


def slice_offsets(positions: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Signed distance of each ImagePositionPatient along the slice normal; exact for any orientation."""
    return positions @ unit_normal(orientation)


def slice_groups(offsets: np.ndarray, tol_mm: float = DUPLICATE_TOL_MM) -> list[list[int]]:
    """Group positions against the first member of each group, so near-duplicates never chain."""
    groups = []
    for i in np.argsort(offsets, kind="stable"):
        if not groups or offsets[i] - offsets[groups[-1][0]] > tol_mm + ROUNDING_MM:
            groups.append([])
        groups[-1].append(int(i))
    return groups


def sort_and_dedupe(offsets: np.ndarray, tol_mm: float = DUPLICATE_TOL_MM) -> SpacingStats:
    """Measure unique positions; callers must validate duplicate image content."""
    order = np.array([g[0] for g in slice_groups(offsets, tol_mm)], dtype=int)
    unique = offsets[order]
    n_duplicates = len(offsets) - len(order)
    if len(unique) < 2:
        return SpacingStats(len(offsets), len(unique), n_duplicates, float("nan"), float("nan"), 0.0, order)
    diffs = np.diff(unique)
    median = float(np.median(diffs))
    return SpacingStats(len(offsets), len(unique), n_duplicates, median, float(np.abs(diffs - median).max()),
                        float(unique[-1] - unique[0]), order)


def is_uniform(stats: SpacingStats, abs_tol_mm: float = 0.05, rel_tol: float = 0.01) -> bool:
    """Every step is within max(abs_tol, rel_tol * median) of the median spacing."""
    if not np.isfinite(stats.spacing_median_mm):
        return False
    return stats.spacing_max_dev_mm <= max(abs_tol_mm, rel_tol * stats.spacing_median_mm) + ROUNDING_MM


# ---- DICOM headers and pixels ----

def dicom_value(ds, tag: str) -> str:
    value = ds.get(tag, "")
    if isinstance(value, (list, pydicom.multival.MultiValue)):
        return "\\".join(str(x) for x in value)
    return "" if value is None else str(value)


def numeric_tag(ds, tag: str, size: int = 1) -> np.ndarray:
    try:
        value = np.atleast_1d(np.asarray(ds.get(tag), dtype=float))
    except (TypeError, ValueError) as exc:
        raise InvalidSeries(f"{tag}_invalid") from exc
    if value.shape != (size,) or not np.isfinite(value).all():
        raise InvalidSeries(f"{tag}_invalid")
    return value


def rescale(ds) -> tuple[float, float]:
    slope = float(numeric_tag(ds, "RescaleSlope")[0])
    intercept = float(numeric_tag(ds, "RescaleIntercept")[0])
    if slope <= 0:
        raise InvalidSeries("RescaleSlope_invalid")
    return slope, intercept


def slice_corners(ds) -> np.ndarray:
    """LPS coordinates (3, 4) of the four corner pixel centres."""
    directions = numeric_tag(ds, "ImageOrientationPatient", 6)
    spacing = numeric_tag(ds, "PixelSpacing", 2)
    xx = np.array([0, ds.Columns - 1, 0, ds.Columns - 1])
    yy = np.array([0, 0, ds.Rows - 1, ds.Rows - 1])
    return (numeric_tag(ds, "ImagePositionPatient", 3)[:, None]
            + np.outer(directions[:3], xx * spacing[1]) + np.outer(directions[3:], yy * spacing[0]))


def stored_values(ds) -> np.ndarray:
    """Pixel samples from the low BitsStored bits; PS3.5 8.1.1 leaves the remaining allocated bits undefined."""
    bits = numeric_tag(ds, "BitsStored")[0]
    if bits != int(bits) or not 0 < bits <= 16:
        raise InvalidSeries("BitsStored_invalid")
    bits = int(bits)
    try:  # pydicom raises many exception classes for malformed, truncated or unsupported pixel data
        values = ds.pixel_array.astype(np.int64) & ((1 << bits) - 1)
    except Exception as exc:
        raise InvalidSeries("pixel_decode_error") from exc
    if numeric_tag(ds, "PixelRepresentation")[0] == 1:  # two's complement within BitsStored
        values -= (values >> (bits - 1)) << bits
    return values


def slice_to_hu(ds) -> np.ndarray:
    slope, intercept = rescale(ds)
    return stored_values(ds).astype(np.float64) * slope + intercept


def dicom_files(directory) -> list[Path]:
    with os.scandir(directory) as entries:
        return sorted(Path(e.path) for e in entries if e.is_file() and e.name.lower().endswith(".dcm"))


def read_headers(files: list[Path]) -> tuple[list, int, str]:
    """Read every header; return the headers, the number of unreadable files and the source-content hash."""
    headers, unread, digest = [], 0, hashlib.sha256()
    for path in files:
        try:  # Any failure while pydicom parses a file is an input defect; the classes it raises are not enumerable.
            digest.update(path.name.encode("utf-8") + b"\0" + file_hash(path).encode("ascii") + b"\n")
            header = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=READ_TAGS)
            list(header)  # Decode deferred values here so that malformed elements belong to this file.
            headers.append(header)
        except Exception:
            unread += 1
    return headers, unread, digest.hexdigest()


def pixel_defects(headers: list) -> int:
    """Number of slices that do not decode to a finite HU array of the declared size."""
    errors = 0
    for header in headers:
        try:
            pixels = slice_to_hu(pydicom.dcmread(header.filename))
            if pixels.shape != (header.Rows, header.Columns) or not np.isfinite(pixels).all():
                raise InvalidSeries("invalid_pixel_array")
        except Exception:
            errors += 1
    return errors


# ---- Series validation ----

def validate_headers(headers: list) -> None:
    for tag in CONSTANT_CONTEXT + CONSTANT_ENCODING:
        if len({dicom_value(h, tag) for h in headers}) != 1:
            raise InvalidSeries(f"{tag}_varies")
    for tag in ("StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID", "SOPInstanceUID"):
        if any(not dicom_value(h, tag) for h in headers):
            raise InvalidSeries(f"{tag}_missing")
    for tag, size in CONSTANT_GEOMETRY.items():
        values = np.array([numeric_tag(h, tag, size) for h in headers])
        if tag != "ImageOrientationPatient" and (values <= 0).any():
            raise InvalidSeries(f"{tag}_invalid")
        if not np.allclose(values, values[0], atol=1e-4, rtol=0):
            raise InvalidSeries(f"{tag}_varies")
    for h in headers:
        for tag in ("Rows", "Columns", "SamplesPerPixel", "BitsAllocated", "BitsStored", "HighBit",
                    "PixelRepresentation"):
            value = numeric_tag(h, tag)[0]
            if value != int(value):
                raise InvalidSeries(f"{tag}_invalid")
        if h.get("Modality") != "CT" or h.get("SOPClassUID") != CT_IMAGE_STORAGE:
            raise InvalidSeries("not_conventional_CT")
        if dicom_value(h, "ImageType").split("\\")[:3] != ["ORIGINAL", "PRIMARY", "AXIAL"]:
            raise InvalidSeries("not_original_primary_axial")
        if str(h.file_meta.get("TransferSyntaxUID", "")) not in TRANSFER_SYNTAXES:
            raise InvalidSeries("unsupported_transfer_syntax")
        if (h.file_meta.get("MediaStorageSOPClassUID") != h.SOPClassUID
                or h.file_meta.get("MediaStorageSOPInstanceUID") != h.SOPInstanceUID):
            raise InvalidSeries("file_meta_SOP_mismatch")
        if dicom_value(h, "NumberOfFrames") not in ("", "1"):
            raise InvalidSeries("multiframe_image")
        if h.SamplesPerPixel != 1 or h.get("PhotometricInterpretation") != "MONOCHROME2":
            raise InvalidSeries("unsupported_pixel_format")
        if (h.BitsAllocated != 16 or not 12 <= h.BitsStored <= 16 or h.HighBit != h.BitsStored - 1
                or h.PixelRepresentation not in (0, 1)):  # CT Image Module: 16 bits allocated, 12-16 stored
            raise InvalidSeries("invalid_pixel_encoding")
        if dicom_value(h, "GantryDetectorTilt") and numeric_tag(h, "GantryDetectorTilt")[0] != 0:
            raise InvalidSeries("gantry_tilt_nonzero")
        if (dicom_value(h, "RescaleType") not in ("", "HU")
                or dicom_value(h, "MultienergyCTAcquisition") not in ("", "NO")
                or "ModalityLUTSequence" in h):
            raise InvalidSeries("unsupported_HU_mapping")
        rescale(h)


def inspect_geometry(headers: list, *, validate: bool = True) -> tuple[SpacingStats, dict]:
    """Measure slice positions; with ``validate`` also enforce the in-plane, spacing and grid tolerances."""
    orientation = numeric_tag(headers[0], "ImageOrientationPatient", 6)
    row, col = orientation[:3], orientation[3:]
    if not np.allclose([np.linalg.norm(row), np.linalg.norm(col), np.dot(row, col)], [1, 1, 0], atol=1e-4, rtol=0):
        raise InvalidSeries("ImageOrientationPatient_invalid")
    positions = np.array([numeric_tag(h, "ImagePositionPatient", 3) for h in headers])
    offsets = slice_offsets(positions, orientation)
    stats = sort_and_dedupe(offsets)
    normal = unit_normal(orientation)
    delta = positions - positions[stats.order[0]]
    lateral = delta - np.outer(delta @ normal, normal)
    stats.inplane_max_dev_mm = float(np.linalg.norm(lateral, axis=1).max())
    details = dict(n_unique_slices=stats.n_unique, n_duplicates=stats.n_duplicates,
                   spacing_median_mm=stats.spacing_median_mm, spacing_max_dev_mm=stats.spacing_max_dev_mm,
                   extent_mm=stats.extent_mm, inplane_max_dev_mm=stats.inplane_max_dev_mm,
                   position_first_lps="\\".join(map(str, positions[stats.order[0]])),
                   position_last_lps="\\".join(map(str, positions[stats.order[-1]])),
                   normal_z=float(abs(normal[2])))
    if not validate:
        return stats, details
    if stats.inplane_max_dev_mm > POSITION_TOL_MM + ROUNDING_MM:
        raise InvalidSeries("inplane_position_varies")
    if not is_uniform(stats):
        raise InvalidSeries("nonuniform_spacing")
    grid_error = float(np.abs(delta[stats.order] @ normal - np.arange(stats.n_unique) * stats.spacing_median_mm).max())
    if grid_error > POSITION_TOL_MM + ROUNDING_MM:
        raise InvalidSeries("slice_positions_not_on_grid")
    try:
        instances = np.array([h.get("InstanceNumber") for h in headers], dtype=float)[stats.order]
        mismatch = not (np.isfinite(instances).all() and np.all(instances == np.rint(instances))
                        and (np.all(np.diff(instances) == 1) or np.all(np.diff(instances) == -1)))
    except (TypeError, ValueError):
        mismatch = True
    details.update(spacing_max_mm=float(np.diff(offsets[stats.order]).max()), grid_max_dev_mm=grid_error,
                   flag_instance_mismatch=mismatch)
    return stats, details


def validate_duplicates(files: list[Path], headers: list) -> None:
    """Slices at one position must share corners and HU; a SOP UID may not recur at another position."""
    positions = np.array([h.ImagePositionPatient for h in headers], dtype=float)
    offsets = slice_offsets(positions, np.array(headers[0].ImageOrientationPatient, dtype=float))
    uid_positions = {}
    for h, position in zip(headers, positions, strict=True):
        previous = uid_positions.setdefault(str(h.SOPInstanceUID), position)
        if not np.allclose(position, previous, atol=DUPLICATE_TOL_MM + ROUNDING_MM, rtol=0):
            raise InvalidSeries("SOPInstanceUID_reused_at_different_position")
    for group in slice_groups(offsets):
        if len(group) < 2:
            continue
        reference = slice_to_hu(pydicom.dcmread(files[group[0]]))
        for i in group[1:]:
            corner_error = slice_corners(headers[i]) - slice_corners(headers[group[0]])
            if np.linalg.norm(corner_error, axis=0).max() > DUPLICATE_TOL_MM + ROUNDING_MM:
                raise InvalidSeries("conflicting_duplicate_geometry")
            if not np.allclose(slice_to_hu(pydicom.dcmread(files[i])), reference, atol=HU_TOL, rtol=0):
                raise InvalidSeries("conflicting_duplicate_pixels")


def audit_series(files: list[Path], pid: str) -> tuple[list, SpacingStats | None, dict]:
    """Inspect every header, then decode every slice of a series that passes the header rules; defects never raise."""
    files = [Path(path) for path in files]
    headers, unread, source_hash = read_headers(files)
    audit = dict.fromkeys(AUDIT_FIELDS, "")
    audit.update(validation_status="rejected", n_unread=unread, source_sha256=source_hash)
    if not headers:
        audit.update(validation_status="incomplete", validation_reason="unreadable_DICOM" if unread else "no_files")
        return headers, None, audit
    patients = {dicom_value(h, "PatientID") for h in headers}
    timepoints = {dicom_value(h, "ClinicalTrialTimePointID") for h in headers}
    audit.update(observed_patient_ids=";".join(sorted(patients)), observed_timepoints=";".join(sorted(timepoints)),
                 varied_tags=";".join(t for t in SUMMARY_TAGS if len({dicom_value(h, t) for h in headers}) > 1),
                 flag_rescale_varies=len({(dicom_value(h, "RescaleSlope"), dicom_value(h, "RescaleIntercept"))
                                         for h in headers}) > 1,
                 flag_acquisition_varies=len({dicom_value(h, "AcquisitionNumber") for h in headers}) > 1)
    stats = None
    try:  # Record the measured range even when another check rejects or holds the series.
        stats, details = inspect_geometry(headers, validate=False)
        audit.update(details)
    except InvalidSeries:
        pass
    identity_error = ""
    for tag in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        values = {dicom_value(h, tag) for h in headers}
        if "" in values or (tag != "SOPInstanceUID" and len(values) != 1):
            identity_error = f"{tag}_missing_or_mixed"
        elif any(not pydicom.uid.UID(value).is_valid for value in values):
            identity_error = f"{tag}_invalid"
    if any(h.file_meta.get("MediaStorageSOPInstanceUID") != h.get("SOPInstanceUID") for h in headers):
        identity_error = "file_meta_SOP_mismatch"
    if patients != {pid} or len(timepoints) != 1 or not timepoints <= set(TIME_POINTS):
        audit.update(validation_status="hold", validation_reason="patient_or_timepoint_conflict")
    elif identity_error:
        audit.update(validation_status="hold", validation_reason=identity_error)
    elif unread:
        audit.update(validation_status="incomplete", validation_reason="unreadable_DICOM")
    else:
        try:
            validate_headers(headers)
            if pixel_defects(headers):
                audit.update(validation_status="incomplete", validation_reason="pixel_decode_error")
            else:
                stats, details = inspect_geometry(headers)
                audit.update(details)
                validate_duplicates(files, headers)
                audit.update(validation_status="valid", validation_reason="")
        except InvalidSeries as exc:
            audit["validation_reason"] = str(exc)
    return headers, stats, audit


# ---- Private inventory index: frozen file listing and all-instance UID audit ----

def listing(directory) -> list:
    files = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith(".dcm"):
                stat = entry.stat()
                files.append((entry.name, stat.st_size, stat.st_mtime_ns))
    return sorted(files, key=lambda item: Path(item[0]))  # dicom_files() order, case-insensitive on Windows


def open_inventory(path, root, jobs, subjects, *, log=None) -> sqlite3.Connection:
    """Freeze all subjects and series before scanning; verify the frozen listing on resume."""
    db = sqlite3.connect(path, timeout=60)  # a concurrent reader must not end a multi-day run
    db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS subjects (pid TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS series (path TEXT PRIMARY KEY, pid TEXT, study TEXT, files TEXT, result TEXT);
        CREATE TABLE IF NOT EXISTS instances (
            path TEXT, file TEXT, pid TEXT, tp TEXT, study_uid TEXT, series_uid TEXT, sop_uid TEXT);
        CREATE INDEX IF NOT EXISTS study_index ON instances(study_uid);
        CREATE INDEX IF NOT EXISTS series_index ON instances(series_uid);
        CREATE INDEX IF NOT EXISTS sop_index ON instances(sop_uid);
        CREATE INDEX IF NOT EXISTS instance_path_index ON instances(path);
    """)
    source = str(Path(root).resolve())
    previous = db.execute("SELECT value FROM metadata WHERE key='root'").fetchone()
    if previous and previous[0] != source:
        raise ValueError("Inventory source changed; use a new output path")
    frozen = db.execute("SELECT value FROM metadata WHERE key='frozen'").fetchone()
    studies = sorted((pid, study.name) for pid in subjects for study in (Path(root) / pid).iterdir() if study.is_dir())
    paths = {Path(directory).relative_to(root).as_posix() for _, _, directory in jobs}
    if frozen:
        old = {row[0] for row in db.execute("SELECT path FROM series")}
        old_subjects = {row[0] for row in db.execute("SELECT pid FROM subjects")}
        old_studies = db.execute("SELECT value FROM metadata WHERE key='studies'").fetchone()
        if old != paths or old_subjects != set(subjects) or old_studies[0] != json.dumps(studies):
            raise ValueError("DICOM listing changed; use a new inventory output")
    with db:
        if not frozen:
            db.executemany("INSERT INTO subjects VALUES (?)", [(pid,) for pid in subjects])
        file_count = 0
        for index, (pid, study, directory) in enumerate(jobs, 1):
            rel = Path(directory).relative_to(root).as_posix()
            files = listing(directory)
            file_count += len(files)
            current = (pid, study, json.dumps(files))
            if frozen:
                if db.execute("SELECT pid,study,files FROM series WHERE path=?", (rel,)).fetchone() != current:
                    raise ValueError("DICOM listing changed; use a new inventory output")
            else:
                db.execute("INSERT INTO series VALUES (?,?,?,?,NULL)", (rel, *current))
            if log is not None and (index % 1000 == 0 or index == len(jobs)):
                log.info("listed files for %d/%d series (%d files)", index, len(jobs), file_count)
        if not frozen:
            db.executemany("INSERT INTO metadata VALUES (?,?)",
                           [("root", source), ("studies", json.dumps(studies)), ("frozen", "yes")])
    return db


def save_series(db, root, row: dict, headers: list) -> None:
    rel = "/".join([row["pid"], row["study_folder"], row["series_folder"]])
    expected = db.execute("SELECT files FROM series WHERE path=?", (rel,)).fetchone()[0]
    if json.dumps(listing(Path(root) / rel)) != expected:
        raise ValueError("DICOM series changed during inspection")
    with db:
        db.execute("DELETE FROM instances WHERE path=?", (rel,))
        db.executemany("INSERT INTO instances VALUES (?,?,?,?,?,?,?)", [
            (rel, Path(h.filename).name, *(dicom_value(h, tag) for tag in
             ("PatientID", "ClinicalTrialTimePointID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")))
            for h in headers])
        serializable = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in row.items()}
        db.execute("UPDATE series SET result=? WHERE path=?", (json.dumps(serializable, allow_nan=False), rel))


def uid_conflicts(db, root) -> set:
    """Series paths whose Study, Series or SOP UIDs have several owners, including instances in rejected series."""
    conflicts = set()
    for uid, owner in (("study_uid", "pid,tp"), ("series_uid", "pid,tp,study_uid"),
                       ("sop_uid", "pid,tp,study_uid,series_uid")):
        query = (f"SELECT {uid} FROM (SELECT DISTINCT {uid},{owner} FROM instances WHERE {uid}!='') "
                 f"GROUP BY {uid} HAVING COUNT(*)>1")
        for (value,) in db.execute(query):
            conflicts.update(p for (p,) in db.execute(f"SELECT DISTINCT path FROM instances WHERE {uid}=?", (value,)))
    # Same-owner copies may share a SOP UID only when their geometry and HU agree.
    for (uid,) in db.execute("SELECT sop_uid FROM instances WHERE sop_uid!='' GROUP BY sop_uid HAVING COUNT(*)>1"):
        copies = db.execute("SELECT path,file FROM instances WHERE sop_uid=?", (uid,)).fetchall()
        try:
            reference = pydicom.dcmread(Path(root, *copies[0]))
            hu = slice_to_hu(reference)
            for path, name in copies[1:]:
                image = pydicom.dcmread(Path(root, path, name))
                corner_error = np.linalg.norm(slice_corners(image) - slice_corners(reference), axis=0).max()
                if (corner_error > DUPLICATE_TOL_MM + ROUNDING_MM
                        or not np.allclose(slice_to_hu(image), hu, atol=HU_TOL, rtol=0)):
                    conflicts.update(p for p, _ in copies)
        except Exception:  # any unreadable copy: the SOP UID cannot be verified, as in pixel_defects
            conflicts.update(p for p, _ in copies)
    return conflicts
