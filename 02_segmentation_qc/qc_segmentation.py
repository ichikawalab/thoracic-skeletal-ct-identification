"""Stage 2: decide from the two masks which examinations and structures enter feature extraction.

Output: --out with 13 rows per examination (sternum, T1-T12): the examination verdict with the failed rules and
each structure's state (usable, absent, truncated, numbering_gap, ineligible_exam). Examination rules, valid for
any chest CT: the sternum and at least one thoracic vertebra are segmented, the sternum centroid is anterior to
the thoracic vertebrae, and the sternum lies within the field of view. Structure rules: a structure of an eligible
examination is usable when it is present and touches neither the first/last slice nor the in-plane border; when a
thoracic label is missing between present ones the numbering is unreliable and no vertebra is used. No feature
value or voxel-count threshold is involved.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import pandas as pd
import SimpleITK as sitk

from common.provenance import (
    SEGMENTATION_FIELDS,
    STRUCTURES,
    THORACIC,
    TIME_POINTS,
    append_records,
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

EXAM_FIELDS = ["sternum_present", "thoracic_present", "sternum_anterior_mm", "sternum_within_fov",
               "exam_eligible", "exam_exclusion", "vertebra_numbering_gap"]
FIELDS = ["pid", "timepoint", "structure", "label", *EXAM_FIELDS, "n_voxels", "touches_z_boundary",
          "touches_xy_boundary", "structure_status", "error", "code_sha256", "status"]
KEYS = ("pid", "timepoint", "structure")


def label_numbers(seg_root: str) -> dict:
    """{structure: label} for the sternum (total task) and T1-T12 (vertebrae_pp_refined task)."""
    with open(os.path.join(seg_root, "label_maps.json"), encoding="utf-8") as stream:
        maps = json.load(stream)
    labels = {name: int(label) for label, name in maps["total"].items() if name == "sternum"}
    labels.update({name: int(label) for label, name in maps["vertebrae_pp_refined"].items() if name in THORACIC})
    if set(labels) != set(STRUCTURES):
        raise KeyError(f"labels not found in label_maps.json: {sorted(set(STRUCTURES) - set(labels))}")
    return labels


def read_mask(path: str) -> tuple[sitk.Image, np.ndarray]:
    """The mask with axes permuted/flipped toward LPS (no interpolation) and its array as [z, y, x]."""
    image = sitk.DICOMOrient(sitk.ReadImage(path), "LPS")
    return image, sitk.GetArrayViewFromImage(image)


def boundary_contact(array: np.ndarray, label: int) -> tuple[bool, bool]:
    """(touches first/last slice, touches in-plane border) for one label; array is [z, y, x]."""
    z = any((face == label).any() for face in (array[0], array[-1]))
    xy = any((face == label).any() for face in (array[:, 0], array[:, -1], array[:, :, 0], array[:, :, -1]))
    return z, xy


def centroid_lps(image: sitk.Image, mask: np.ndarray) -> np.ndarray:
    """Physical (LPS) centroid of the true voxels of ``mask``, which is [z, y, x] for ``image``."""
    z, y, x = np.argwhere(mask).mean(axis=0)
    return np.array(image.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z))))


def qc_exam(seg_root: str, exam: dict, labels: dict) -> list:
    """Thirteen rows: the examination verdict repeated, and each structure's own state."""
    rows = [{**dict.fromkeys(FIELDS, ""), "pid": exam["pid"], "timepoint": exam["timepoint"], "structure": name,
             "label": labels[name]} for name in STRUCTURES]
    try:
        sternum_image, sternum = read_mask(os.path.join(seg_root, exam["sternum_path"]))
        vertebrae_image, vertebrae = read_mask(os.path.join(seg_root, exam["vertebrae_path"]))
    except RuntimeError as e:
        for row in rows:
            row.update(error=f"read_error: {e}"[:200], status="error")
        return rows
    sternum_mask = sternum == labels["sternum"]
    present = [(vertebrae == labels[name]).any() for name in THORACIC]
    thoracic_mask = np.isin(vertebrae, [labels[name] for name in THORACIC])
    verdict = dict(sternum_present=bool(sternum_mask.any()), thoracic_present=any(present))
    exclusion = [key for key in ("sternum_present", "thoracic_present") if not verdict[key]]
    if not exclusion:
        # LPS: y grows posteriorly, so a positive difference places the sternum anterior to the vertebrae.
        anterior = centroid_lps(vertebrae_image, thoracic_mask)[1] - centroid_lps(sternum_image, sternum_mask)[1]
        verdict["sternum_anterior_mm"] = round(float(anterior), 1)
        if anterior <= 0:
            exclusion.append("sternum_anterior")
    verdict["sternum_within_fov"] = not any(boundary_contact(sternum, labels["sternum"]))
    if not verdict["sternum_within_fov"]:
        exclusion.append("sternum_within_fov")
    # A thoracic label missing between present ones means the numbering skipped or merged a vertebra.
    gap = any(not present[i] and any(present[:i]) and any(present[i + 1:]) for i in range(len(present)))
    verdict.update(exam_eligible=not exclusion, exam_exclusion=";".join(exclusion), vertebra_numbering_gap=gap)
    for row in rows:
        array = sternum if row["structure"] == "sternum" else vertebrae
        n_voxels = int((array == row["label"]).sum())
        z, xy = boundary_contact(array, row["label"]) if n_voxels else (False, False)
        if exclusion:
            state = "ineligible_exam"
        elif gap and row["structure"] != "sternum":
            state = "numbering_gap"
        elif not n_voxels:
            state = "absent"
        else:
            state = "truncated" if z or xy else "usable"
        row.update(verdict, n_voxels=n_voxels, touches_z_boundary=z, touches_xy_boundary=xy, structure_status=state,
                   status="complete")
    return rows


def resume_qc(path: str, expected_exams: set) -> set:
    """Examinations whose thirteen latest records are complete; the checkpoint is compacted to the latest records."""
    if not os.path.exists(path):
        return set()
    latest = read_manifest(path, FIELDS, KEYS)
    if not set(zip(latest["pid"], latest["timepoint"], strict=True)) <= expected_exams:
        raise ValueError("QC checkpoint differs from segmentation inputs")
    write_csv_atomic(latest, path)
    complete = latest[latest["status"].eq("complete") & latest["error"].eq("")]
    counts = complete.groupby(["pid", "timepoint"]).size()
    return set(counts[counts == len(STRUCTURES)].index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seg-root", required=True, help="directory with segmentation_manifest.csv and label_maps.json")
    ap.add_argument("--out", required=True, help="output CSV: one row per examination and structure")
    ap.add_argument("--timepoints", default="T0,T2")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N examinations (trial run)")
    args = ap.parse_args()
    if args.workers < 1 or args.limit < 0 or not set(args.timepoints.split(",")) <= set(TIME_POINTS):
        ap.error("Invalid workers, limit or timepoints")
    with output_lock(args.out):
        run_qc(args)


def run_qc(args: argparse.Namespace) -> None:
    input_manifest = os.path.join(args.seg_root, "segmentation_manifest.csv")
    require_completion(input_manifest)
    log = start_run(__file__, args.out, args, {"structures": STRUCTURES})
    labels = label_numbers(args.seg_root)
    seg = read_manifest(input_manifest, SEGMENTATION_FIELDS)
    seg = seg[seg["timepoint"].isin(args.timepoints.split(","))].to_dict("records")
    if not all(verified_file(args.seg_root, exam, name) for exam in seg for name in ("sternum", "vertebrae")):
        raise ValueError("Mask is missing or changed; resume segmentation first")
    expected_exams = {(r["pid"], r["timepoint"]) for r in seg}
    done = resume_qc(args.out, expected_exams)
    todo = [exam for exam in seg if (exam["pid"], exam["timepoint"]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    log.info("segmented examinations: %d, already assessed: %d, to assess: %d", len(seg), len(done), len(todo))
    code = source_hash()
    t0, n_err = time.perf_counter(), 0
    with worker_pools(log) as pools, append_records(args.out, FIELDS) as write:
        pool = ProcessPoolExecutor(args.workers)
        pools.append(pool)
        for i, rows in enumerate(pool.map(qc_exam, repeat(args.seg_root), todo, repeat(labels), chunksize=4), 1):
            for row in rows:
                row["code_sha256"] = code
            write(rows)
            n_err += bool(rows[0]["error"])
            if i % 500 == 0 or i == len(todo):
                elapsed = time.perf_counter() - t0
                log.info("%d/%d exams  %.1f min  (%.2f exam/s)  read errors=%d", i, len(todo), elapsed / 60,
                         i / elapsed, n_err)
    log.info("finished: %d examinations, %d read errors", len(todo), n_err)
    expected_keys = {(pid, timepoint, name) for pid, timepoint in expected_exams for name in STRUCTURES}
    complete = finish_manifest(args.out, FIELDS, expected_keys, KEYS, limited=bool(args.limit))
    if complete:
        table = pd.read_csv(args.out, dtype=str, keep_default_na=False)
        exams = table.drop_duplicates(["pid", "timepoint"])
        log.info("examinations: %d eligible, %d ineligible %s", exams["exam_eligible"].eq("True").sum(),
                 exams["exam_eligible"].ne("True").sum(),
                 exams.loc[exams["exam_exclusion"].ne(""), "exam_exclusion"].value_counts().to_dict())
        log.info("structures: %s", table.groupby("structure")["structure_status"].value_counts().unstack(fill_value=0)
                 .reindex(STRUCTURES).to_string())
    if n_err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
