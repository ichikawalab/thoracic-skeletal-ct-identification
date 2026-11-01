"""Stage 3: extract the 14 PyRadiomics shape features of every usable structure (env-radiomics, Python 3.9).

Output: --out with one row per examination and usable structure (from the QC table): the features, seconds and
status (computed or error). Masks are reordered toward LPS and resampled to 1 mm isotropic spacing with the
settings in pyradiomics_shape.yaml; the mask serves as both image and label image, so only geometry enters.
Resumes per structure.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import repeat

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Python 3.9: common is not installed
from common.provenance import (  # noqa: E402
    SEGMENTATION_FIELDS,
    SHAPE_FEATURES,
    TIME_POINTS,
    append_records,
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

FIELDS = ["pid", "timepoint", "structure", "label", *SHAPE_FEATURES, "seconds", "error", "code_sha256", "status"]
KEYS = ("pid", "timepoint", "structure")
STATUSES = ("computed", "error")


@lru_cache(maxsize=1)
def get_extractor(params_path: str):
    import radiomics
    from radiomics import featureextractor

    radiomics.logger.setLevel(logging.ERROR)
    return featureextractor.RadiomicsFeatureExtractor(params_path)


def structure_row(pid: str, tp: str, name: str, label: int, image: sitk.Image, params_path: str) -> dict:
    t0 = time.perf_counter()
    row = dict.fromkeys(FIELDS, "")
    row.update(pid=pid, timepoint=tp, structure=name, label=label, status="error")
    try:
        result = get_extractor(params_path).execute(image, image, label=label)
        features = {feat: float(result[f"original_shape_{feat}"]) for feat in SHAPE_FEATURES}
        if not np.isfinite(list(features.values())).all():
            raise ValueError("Nonfinite shape features")
        row.update(features, status="computed")
    except (ValueError, RuntimeError) as e:
        row["error"] = f"{type(e).__name__}: {e}"[:200]
    row["seconds"] = round(time.perf_counter() - t0, 2)
    return row


def extract_exam(seg_root: str, exam: dict, params_path: str) -> list:
    """Rows for ``exam["structures"]`` ({name: label}); the sternum comes from its own mask, T1-T12 from the other."""
    by_mask = {"sternum_path": [], "vertebrae_path": []}
    for name in exam["structures"]:
        by_mask["sternum_path" if name == "sternum" else "vertebrae_path"].append(name)
    rows = []
    for mask_key, names in by_mask.items():
        if not names:
            continue
        try:
            image = sitk.DICOMOrient(sitk.ReadImage(os.path.join(seg_root, exam[mask_key])), "LPS")
        except (OSError, RuntimeError) as e:
            rows += [{**dict.fromkeys(FIELDS, ""), "pid": exam["pid"], "timepoint": exam["timepoint"],
                      "structure": name, "label": exam["structures"][name], "error": f"read_error: {e}"[:200],
                      "status": "error"} for name in names]
            continue
        rows += [structure_row(exam["pid"], exam["timepoint"], name, exam["structures"][name], image, params_path)
                 for name in names]
    return rows


def resume_features(path: str, expected_keys: set) -> set:
    """Structures whose latest record is computed with finite features; the checkpoint is compacted to those records."""
    if not os.path.exists(path):
        return set()
    latest = read_manifest(path, FIELDS, KEYS, STATUSES)
    actual = set(latest[list(KEYS)].itertuples(index=False, name=None))
    if not actual <= expected_keys:
        raise ValueError("Feature checkpoint differs from the usable structures")
    features = latest[SHAPE_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    done = latest["error"].eq("") & latest["status"].eq("computed") & np.isfinite(features).all(axis=1)
    write_csv_atomic(latest, path)
    return set(latest.loc[done, list(KEYS)].itertuples(index=False, name=None))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seg-root", required=True, help="directory with segmentation_manifest.csv")
    ap.add_argument("--qc", required=True, help="segmentation_qc.csv written by the QC stage")
    ap.add_argument("--params", required=True, help="PyRadiomics parameter file (YAML)")
    ap.add_argument("--out", required=True, help="output CSV: one row per examination and usable structure")
    ap.add_argument("--timepoints", default="T0,T2")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N examinations (trial run)")
    args = ap.parse_args()
    if args.workers < 1 or args.limit < 0 or not set(args.timepoints.split(",")) <= set(TIME_POINTS):
        ap.error("Invalid workers, limit or timepoints")
    with output_lock(args.out):
        run_extraction(args)


def run_extraction(args: argparse.Namespace) -> None:
    input_manifest = os.path.join(args.seg_root, "segmentation_manifest.csv")
    require_completion(input_manifest)
    qc_inputs = require_completion(args.qc)["run_identity"]["inputs"]
    if qc_inputs.get("segmentation_manifest.csv") != file_hash(input_manifest):
        raise ValueError("QC table was made from a different segmentation manifest")
    with open(args.params, encoding="utf-8") as stream:
        params_text = stream.read()
    log = start_run(__file__, args.out, args, {"params_file": os.path.basename(args.params), "params": params_text,
                                              "features": SHAPE_FEATURES})
    get_extractor(args.params)  # invalid settings stop the run before any worker starts
    timepoints = args.timepoints.split(",")
    seg = read_manifest(input_manifest, SEGMENTATION_FIELDS)
    seg = seg[seg["timepoint"].isin(timepoints)].to_dict("records")
    if not all(verified_file(args.seg_root, exam, name) for exam in seg for name in ("sternum", "vertebrae")):
        raise ValueError("Mask is missing or changed; resume segmentation first")
    qc = pd.read_csv(args.qc, dtype=str, keep_default_na=False)
    qc = qc[qc["timepoint"].isin(timepoints)]
    if {(r.pid, r.timepoint) for r in qc.itertuples()} != {(exam["pid"], exam["timepoint"]) for exam in seg}:
        raise ValueError("QC table does not cover exactly the segmented examinations of the requested time points")
    structures: dict[tuple, dict] = {}
    for r in qc[qc["structure_status"].eq("usable")].itertuples():
        structures.setdefault((r.pid, r.timepoint), {})[r.structure] = int(r.label)
    expected_keys = {(pid, tp, name) for (pid, tp), names in structures.items() for name in names}
    done = resume_features(args.out, expected_keys)
    todo = []
    for exam in seg:
        remaining = {name: label for name, label in structures.get((exam["pid"], exam["timepoint"]), {}).items()
                     if (exam["pid"], exam["timepoint"], name) not in done}
        if remaining:
            todo.append({**exam, "structures": remaining})
    if args.limit:
        todo = todo[:args.limit]
    log.info("usable structures: %d in %d examinations, completed: %d, examinations to process: %d",
             len(expected_keys), len(structures), len(done), len(todo))
    code = source_hash()
    t0, n_err = time.perf_counter(), 0
    with worker_pools(log) as pools, append_records(args.out, FIELDS) as write:
        pool = ProcessPoolExecutor(args.workers)
        pools.append(pool)
        jobs = pool.map(extract_exam, repeat(args.seg_root), todo, repeat(args.params), chunksize=4)
        for i, rows in enumerate(jobs, 1):
            for row in rows:
                row["code_sha256"] = code
            write(rows)
            n_err += sum(bool(row["error"]) for row in rows)
            if i % 500 == 0 or i == len(todo):
                elapsed = time.perf_counter() - t0
                log.info("%d/%d exams  %.1f min  (%.2f exam/s)  structure errors=%d",
                         i, len(todo), elapsed / 60, i / elapsed, n_err)
    log.info("finished: %d examinations, %d structure errors", len(todo), n_err)
    finish_manifest(args.out, FIELDS, expected_keys, KEYS, STATUSES, limited=bool(args.limit))
    if n_err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
