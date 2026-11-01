"""Stage 1c: convert each selected series to one NIfTI volume with dcm2niix and verify it against the source.

Output (--out-root): <pid>/<pid>_<timepoint>.nii.gz in native geometry and HU, nifti_manifest.csv (one row per
examination with geometry, verification result and file hash) and the dcm2niix log of any failed conversion.
Every slice of the published volume must match the DICOM corner coordinates within 0.05 mm and the HU values
within 0.01; series with duplicate positions, mixed pixel encodings or rounded positions are converted from a
corrected temporary copy.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import dcm2niix
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom

from common.dicom_series import (
    HU_TOL,
    POSITION_TOL_MM,
    ROUNDING_MM,
    InvalidSeries,
    audit_series,
    dicom_files,
    slice_corners,
    slice_to_hu,
    stored_values,
    unit_normal,
)
from common.provenance import (
    NIFTI_FIELDS as FIELDS,
    TIME_POINTS,
    append_records,
    atomic_path,
    completed_exams,
    file_hash,
    finish_manifest,
    output_lock,
    read_manifest,
    require_completion,
    source_hash,
    start_run,
    verified_file,
    worker_pools,
    write_csv_atomic,
)

DCM2NIIX = dcm2niix.bin + (".exe" if os.name == "nt" else "")
# Ignore user defaults (-g i), one series per directory (-a y), no subdirectory search (-d 0), no BIDS sidecar
# (-b n), merge slices (-m y), Philips scaling (-p y), original 16-bit scaling (-l o), no cropping or rotation
# (-x i), internal gzip at level 1 (-z i -1), single output name (-f volume).
OPTIONS = ["-g", "i", "-a", "y", "-d", "0", "-b", "n", "-m", "y", "-p", "y",
           "-l", "o", "-x", "i", "-z", "i", "-1", "-f", "volume"]


def converter_version() -> str:
    result = subprocess.run([DCM2NIIX, "--version"], text=True, capture_output=True, check=False)
    if result.returncode not in (0, 3):  # dcm2niix returns 3 after reporting its version
        result.check_returncode()
    versions = re.findall(r"^v\d+\.\d+\.\d+\w*$", result.stdout, flags=re.MULTILINE)
    if len(versions) != 1:
        raise ValueError("Unrecognized dcm2niix version report")
    return versions[0]


def reference_affine(first: pydicom.Dataset, slice_spacing: float) -> np.ndarray:
    """RAS affine expected from the first slice's DICOM geometry."""
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    row, col = orientation[:3], orientation[3:]
    affine = np.eye(4)
    affine[:3, 0] = row * float(first.PixelSpacing[1])
    affine[:3, 1] = col * float(first.PixelSpacing[0])
    affine[:3, 2] = unit_normal(orientation) * slice_spacing
    affine[:3, 3] = first.ImagePositionPatient
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine


def grid_positions(first: pydicom.Dataset, spacing: float, n: int) -> np.ndarray:
    """LPS positions (n, 3) of the retained slices on the median-spacing grid through the first slice."""
    normal = unit_normal(np.asarray(first.ImageOrientationPatient, dtype=float))
    return np.asarray(first.ImagePositionPatient, dtype=float) + np.outer(np.arange(n) * spacing, normal)


def validate_volume(path: Path, files: list[Path], headers: list, stats) -> dict:
    """Compare every slice's corner coordinates and HU with the DICOM source."""
    first = headers[stats.order[0]]
    expected = reference_affine(first, stats.spacing_median_mm)
    image = nib.load(path)
    if not np.isfinite(image.affine).all():
        raise InvalidSeries("NIfTI_affine_nonfinite")
    orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine),
                                                  nib.orientations.io_orientation(expected))
    aligned = image.as_reoriented(orientation)
    if aligned.shape != (int(first.Columns), int(first.Rows), stats.n_unique):
        raise InvalidSeries("NIfTI_shape_mismatch")
    data = np.asanyarray(aligned.dataobj)
    if not np.isfinite(data).all():
        raise InvalidSeries("NIfTI_nonfinite_HU")
    xx = np.array([0, first.Columns - 1, 0, first.Columns - 1])
    yy = np.array([0, 0, first.Rows - 1, first.Rows - 1])
    max_position_error = 0.0
    for k, i in enumerate(stats.order):
        corners = slice_corners(headers[i]) * np.array([-1, -1, 1])[:, None]  # LPS to RAS
        actual = (aligned.affine @ np.vstack([xx, yy, np.full(4, k), np.ones(4)]))[:3]
        error = float(np.linalg.norm(actual - corners, axis=0).max())
        max_position_error = max(max_position_error, error)
        if error > POSITION_TOL_MM + ROUNDING_MM:
            raise InvalidSeries(f"NIfTI_corner_positions_mismatch_slice_{k}")
        if not np.allclose(data[:, :, k], slice_to_hu(pydicom.dcmread(files[i])).T, atol=HU_TOL, rtol=0):
            raise InvalidSeries(f"NIfTI_HU_mismatch_slice_{k}")
    return {"rows": int(first.Rows), "cols": int(first.Columns),
            "voxel_x_mm": float(first.PixelSpacing[1]), "voxel_y_mm": float(first.PixelSpacing[0]),
            "voxel_z_mm": stats.spacing_median_mm, "hu_min": float(data.min()), "hu_max": float(data.max()),
            "max_position_error_mm": max_position_error}


def uniform_encoding(ds: pydicom.Dataset) -> pydicom.Dataset:
    """Rewrite the stored values as 16-bit signed integers with the same HU mapping.

    dcm2niix applies the first slice's bit depth and signedness to a series with one rescale, so slices that differ
    in encoding are converted from copies that agree; the published volume is verified against the original slices.
    """
    values = stored_values(ds)
    if values.max() > 32767:  # unsigned values above the signed range move into the intercept
        values -= 32768
        ds.RescaleIntercept = float(ds.RescaleIntercept) + 32768 * float(ds.RescaleSlope)
    ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 15, 1
    ds.PixelData = values.astype(np.int16).tobytes()
    return ds


def convert_series(root: str, relpath: str, out_path: str, pid: str, timepoint: str,
                   expected_uid: str | None = None, staging_dir: str | None = None,
                   expected_source: str | None = None) -> dict:
    """Re-validate the series, convert it in a temporary directory and publish one verified volume."""
    t0 = time.monotonic()
    source_dir = Path(root, relpath).resolve()
    if Path(root).resolve() not in source_dir.parents:
        raise InvalidSeries("source_path_outside_root")
    output = Path(out_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(str(output) + ".dcm2niix.log")  # kept only when the conversion fails
    log_path.unlink(missing_ok=True)
    files = dicom_files(source_dir)
    headers, stats, audit = audit_series(files, pid)
    if audit["validation_status"] != "valid":
        raise InvalidSeries(audit["validation_reason"])
    if audit["observed_timepoints"] != timepoint:
        raise InvalidSeries("ClinicalTrialTimePointID_mismatch")
    if expected_uid is not None and str(headers[0].SeriesInstanceUID) != expected_uid:
        raise InvalidSeries("SeriesInstanceUID_mismatch")
    if expected_source is not None and audit["source_sha256"] != expected_source:
        raise InvalidSeries("source_changed_since_inventory")
    staging = Path(staging_dir).resolve() if staging_dir else output.parent
    staging.mkdir(parents=True, exist_ok=True)
    mixed = len({(int(h.BitsStored), int(h.PixelRepresentation)) for h in headers}) > 1
    # dcm2niix takes the slice pitch from one adjacent pair and anchors the volume at one slice, so rounded
    # positions can drift by up to (n - 1) x deviation; such series are converted from copies whose positions
    # lie on the median-spacing grid. The published volume is verified against the original slices as always.
    snap = (stats.n_unique - 1) * stats.spacing_max_dev_mm + audit["grid_max_dev_mm"] > POSITION_TOL_MM
    with tempfile.TemporaryDirectory(dir=staging, prefix="dcm2niix-") as work:
        work = Path(work)
        if stats.n_duplicates or mixed or snap:  # one representative per position, uniformly encoded, on the grid
            source_dir = work / "dicom"
            source_dir.mkdir()
            positions = grid_positions(headers[stats.order[0]], stats.spacing_median_mm, stats.n_unique)
            for k, i in enumerate(stats.order):
                if mixed or snap:
                    ds = uniform_encoding(pydicom.dcmread(files[i])) if mixed else pydicom.dcmread(files[i])
                    if snap:
                        ds.ImagePositionPatient = [f"{v:.6f}" for v in positions[k]]
                    ds.save_as(source_dir / files[i].name, write_like_original=False)
                else:
                    shutil.copyfile(files[i], source_dir / files[i].name)
        result = subprocess.run([DCM2NIIX, *OPTIONS, "-o", str(work), str(source_dir)], text=True,
                                capture_output=True, check=False)
        volumes = list(work.glob("*.nii*"))
        try:
            if result.returncode:
                raise InvalidSeries(f"dcm2niix_exit_{result.returncode}")
            if len(volumes) != 1 or not volumes[0].name.endswith(".nii.gz"):
                raise InvalidSeries(f"Expected one NIfTI volume, found {len(volumes)}")
            details = validate_volume(volumes[0], files, headers, stats)
        except Exception:  # any failure to read or verify the output keeps the converter's log for diagnosis
            log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
            raise
        with atomic_path(output) as temporary:
            shutil.copyfile(volumes[0], temporary)
    return {**details, "n_files": len(files), "n_unique_slices": stats.n_unique, "n_duplicates": stats.n_duplicates,
            "seconds": round(time.monotonic() - t0, 1)}


def convert_exam(root: str, out_root: str, exam: dict, staging_dir: str | None = None) -> dict:
    row = dict.fromkeys(FIELDS, "")
    row.update(pid=exam["pid"], timepoint=exam["timepoint"], series_relpath=exam["series_relpath"],
               SeriesInstanceUID=exam["SeriesInstanceUID"], source_sha256=exam["source_sha256"])
    relative = f"{exam['pid']}/{exam['pid']}_{exam['timepoint']}.nii.gz"
    out_path = os.path.join(out_root, relative)
    try:
        row.update(convert_series(root, exam["series_relpath"], out_path, exam["pid"], exam["timepoint"],
                                  exam["SeriesInstanceUID"], staging_dir, exam["source_sha256"]))
        row.update(nifti_path=relative, nifti_sha256=file_hash(out_path), status="complete")
    except (InvalidSeries, subprocess.CalledProcessError, OSError, nib.filebasedimages.ImageFileError) as e:
        row.update(error=f"{type(e).__name__}: {e}"[:200], status="error")
    if os.path.exists(out_path + ".dcm2niix.log"):
        row["dcm2niix_log"] = relative + ".dcm2niix.log"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="DICOM root")
    ap.add_argument("--selection-dir", required=True, help="directory containing exam_selection.csv")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--timepoints", default="T0,T2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--staging-dir", help="temporary conversion directory on a fast local drive")
    ap.add_argument("--limit", type=int, default=0, help="convert only the first N examinations (trial run)")
    args = ap.parse_args()
    if args.workers < 1 or args.limit < 0 or not set(args.timepoints.split(",")) <= set(TIME_POINTS):
        ap.error("Invalid workers, limit or timepoints")
    with output_lock(os.path.join(args.out_root, "nifti_manifest.csv")):
        run_conversion(args)


def run_conversion(args: argparse.Namespace) -> None:
    require_completion(os.path.join(args.selection_dir, "exam_selection.csv"))
    version = converter_version()
    manifest_path = os.path.join(args.out_root, "nifti_manifest.csv")
    log = start_run(__file__, manifest_path, args,
                    {"dcm2niix_version": version, "dcm2niix_sha256": file_hash(DCM2NIIX), "dcm2niix_options": OPTIONS})
    selected = pd.read_csv(os.path.join(args.selection_dir, "exam_selection.csv"), dtype=str, keep_default_na=False)
    selected = selected[selected["status"].eq("selected") & selected["timepoint"].isin(args.timepoints.split(","))]
    exams = selected.to_dict("records")
    done = set()
    if os.path.exists(manifest_path):  # start_run refused to resume if the selection or the time points changed
        previous = read_manifest(manifest_path, FIELDS)
        done = completed_exams([r for r in previous.to_dict("records") if verified_file(args.out_root, r, "nifti")])
        write_csv_atomic(previous, manifest_path)
    todo = [r for r in exams if (r["pid"], r["timepoint"]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    log.info("selected examinations: %d, already converted: %d, todo: %d", len(exams), len(done), len(todo))
    code = source_hash()
    t0, n_err = time.monotonic(), 0
    with worker_pools(log) as pools, append_records(manifest_path, FIELDS) as write:
        pool = ProcessPoolExecutor(args.workers)
        pools.append(pool)
        results = pool.map(convert_exam, repeat(args.root), repeat(args.out_root), todo, repeat(args.staging_dir),
                           chunksize=1)
        for i, row in enumerate(results, 1):
            row["code_sha256"] = code
            write([row])
            n_err += bool(row["error"])
            if i % 200 == 0 or i == len(todo):
                log.info("%d/%d exams  %.1f min  errors=%d", i, len(todo), (time.monotonic() - t0) / 60, n_err)
    log.info("finished: %d converted, %d errors", len(todo) - n_err, n_err)
    finish_manifest(manifest_path, FIELDS, {(r["pid"], r["timepoint"]) for r in exams}, limited=bool(args.limit))
    if n_err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
