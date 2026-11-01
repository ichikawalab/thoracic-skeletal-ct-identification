"""Rank every query (T2) examination against the gallery (T0) by the shape similarity of the structures both share.

Input: the completed shape-feature table, the QC table and the selection, NIfTI and segmentation directories, whose
completion records tie the features to the segmentation, the segmentation to the converted volumes and the volumes to
the selection. Output (--out-dir): cohort.csv (role and usable
structures per eligible examination), matching_results.csv (per query and method: shared structures, scored
candidates, own-T0 score and rank, top candidate and two highest scores), query_structures.csv (per-structure
similarities to the own T0 and to the integrated rank-1 impostor, renumbered vertebrae scores) and
impostor_histogram.csv (all other query-gallery scores).

Roles follow from eligibility: an examination takes part when at least MIN_USABLE structures are usable (the
sternum and at least one vertebra with reliable numbering); every such T0 examination is a gallery entry, and every
such T2 examination whose own T0 is in the gallery is a query. Features are standardized with gallery statistics,
each shared structure gives 1 / (1 + Euclidean distance), and a method's score is the mean over its structures
(sternum only, vertebrae only, integrated). The rank counts every gallery score at least as high as the own T0's,
so ties count against the method; a pair without a shared structure has no score. No exclusion uses the features.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from common.provenance import (
    SHAPE_FEATURES as FEATURES,
    STRUCTURES,
    THORACIC,
    file_hash,
    mark_completion,
    output_lock,
    require_completion,
    start_run,
    write_csv_atomic,
)

METHODS = {"sternum": ["sternum"], "vertebrae": THORACIC, "integrated": STRUCTURES}
ACQUISITION = ["manufacturer_group", "ManufacturerModelName", "kernel_class", "thickness_mm"]
N_BINS = 100_000  # impostor scores are kept as a histogram over [0, 1]: threshold resolution 1e-5
MIN_USABLE = 2  # a single-structure score is not comparable with a mean over many structures
RESULT_FIELDS = ["pid", "method", "n_common", "n_scored", "true_score", "true_rank", "top1_pid", "top1_score",
                 "top2_score"]


def assign_roles(qc: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """One row per eligible examination: role, usable structures and acquisition fields of its selected series."""
    eligible = qc[qc["exam_eligible"].eq("True") & qc["structure_status"].eq("usable")]
    cohort = eligible.groupby(["pid", "timepoint"], sort=True)["structure"].agg(";".join)
    cohort = cohort.rename("usable_structures").reset_index()  # an eligible examination always has a usable sternum
    cohort["n_usable"] = cohort["usable_structures"].str.count(";") + 1
    enough = cohort["n_usable"] >= MIN_USABLE
    gallery = set(cohort.loc[enough & cohort["timepoint"].eq("T0"), "pid"])
    cohort["role"] = np.select([~enough, cohort["timepoint"].eq("T0"), cohort["pid"].isin(gallery)],
                               ["single_structure", "gallery", "query"], "no_reference")
    selected = selection.loc[selection["status"].eq("selected"), ["pid", "timepoint", *ACQUISITION]]
    cohort = cohort.merge(selected, on=["pid", "timepoint"], how="left", validate="one_to_one")
    if cohort[ACQUISITION].isna().any().any():
        raise ValueError("Eligible examination without a selected series")
    return cohort[["pid", "timepoint", "role", "n_usable", "usable_structures", *ACQUISITION]]


def feature_arrays(features: pd.DataFrame, pids: np.ndarray, timepoint: str) -> dict:
    """{structure: (len(pids), 14) array}; the row of an examination without that usable structure is NaN."""
    rows = features[features["timepoint"].eq(timepoint) & features["pid"].isin(pids)]
    position = pd.Series(np.arange(len(pids)), index=pids)
    arrays = {}
    for name in STRUCTURES:
        part = rows[rows["structure"].eq(name)]
        array = np.full((len(pids), len(FEATURES)), np.nan)
        array[position[part["pid"]].to_numpy()] = part[FEATURES].to_numpy(dtype=float)
        arrays[name] = array
    return arrays


def gallery_scaling(gallery: dict) -> dict:
    """{structure: (mean, sd)} over the gallery examinations that have the structure (sd with n - 1)."""
    scaling = {}
    for name, array in gallery.items():
        present = array[np.isfinite(array[:, 0])]
        if len(present) == 0:  # a structure no gallery examination has is shared with no pair
            scaling[name] = (np.full(len(FEATURES), np.nan), np.full(len(FEATURES), np.nan))
            continue
        mean, sd = present.mean(axis=0), present.std(axis=0, ddof=1)
        if not (np.isfinite(sd).all() and (sd > 0).all()):
            raise ValueError(f"Constant or undefined feature in the gallery for {name}")
        scaling[name] = (mean, sd)
    return scaling


def standardize(arrays: dict, scaling: dict) -> dict:
    return {name: (array - scaling[name][0]) / scaling[name][1] for name, array in arrays.items()}


def structure_similarity(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """1 / (1 + Euclidean distance) for every query-gallery pair; NaN where either examination lacks the structure."""
    similarity = np.full((len(query), len(gallery)), np.nan)
    q_ok, g_ok = np.isfinite(query[:, 0]), np.isfinite(gallery[:, 0])
    similarity[np.ix_(q_ok, g_ok)] = 1 / (1 + cdist(query[q_ok], gallery[g_ok]))
    return similarity


def pair_similarity(query: np.ndarray, gallery: np.ndarray, q_idx: np.ndarray, g_idx: np.ndarray) -> np.ndarray:
    """Similarity of the given query-gallery pairs (NaN propagates from a missing structure)."""
    return 1 / (1 + np.linalg.norm(query[q_idx] - gallery[g_idx], axis=1))


def mean_score(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def shifted_vertebrae_scores(query_raw: dict, gallery: dict, scaling: dict, mate: np.ndarray) -> dict:
    """Vertebrae-only score to the mate after renumbering the query's vertebrae by one level in each direction.

    ``shift_up`` compares query T_k with the mate's T_(k+1), ``shift_down`` with T_(k-1); each query level is
    standardized with the gallery statistics of the level it is compared with.
    """
    scores = {}
    for label, offset in (("vertebrae_shift_up", 1), ("vertebrae_shift_down", -1)):
        total, count = np.zeros(len(mate)), np.zeros(len(mate), dtype=int)
        for k in range(1, 13):
            if not 1 <= k + offset <= 12:
                continue
            target = f"vertebrae_T{k + offset}"
            mean, sd = scaling[target]
            query = (query_raw[f"vertebrae_T{k}"] - mean) / sd
            similarity = pair_similarity(query, gallery[target], np.arange(len(mate)), mate)
            finite = np.isfinite(similarity)
            total[finite] += similarity[finite]
            count += finite
        scores[label] = mean_score(total, count)
    return scores


def match(features: pd.DataFrame, cohort: pd.DataFrame, chunk: int = 500) -> dict:
    """Score, rank and summarize every query against the whole gallery; returns the output tables and timing."""
    gallery_pids = np.sort(cohort.loc[cohort["role"].eq("gallery"), "pid"].to_numpy())
    query_pids = np.sort(cohort.loc[cohort["role"].eq("query"), "pid"].to_numpy())
    gallery_raw, query_raw = feature_arrays(features, gallery_pids, "T0"), feature_arrays(features, query_pids, "T2")
    scaling = gallery_scaling(gallery_raw)
    gallery, query = standardize(gallery_raw, scaling), standardize(query_raw, scaling)
    mate = np.searchsorted(gallery_pids, query_pids)  # gallery position of each query's own T0 examination
    histogram = {method: np.zeros(N_BINS, dtype=np.int64) for method in METHODS}
    mate_similarity = {name: np.full(len(query_pids), np.nan) for name in STRUCTURES}  # the values scored, exactly
    impostor_idx = np.zeros(len(query_pids), dtype=int)
    records = {method: [] for method in METHODS}
    started = time.perf_counter()
    for start in range(0, len(query_pids), chunk):
        rows = slice(start, min(start + chunk, len(query_pids)))
        own = np.arange(rows.stop - rows.start)
        sternum = structure_similarity(query["sternum"][rows], gallery["sternum"])
        mate_similarity["sternum"][rows] = sternum[own, mate[rows]]
        total, count = np.zeros_like(sternum), np.zeros(sternum.shape, dtype=np.int16)
        for name in THORACIC:
            similarity = structure_similarity(query[name][rows], gallery[name])
            mate_similarity[name][rows] = similarity[own, mate[rows]]
            finite = np.isfinite(similarity)
            total[finite] += similarity[finite]
            count += finite
        scores = {"sternum": sternum, "vertebrae": mean_score(total, count)}
        counts = {"sternum": np.isfinite(sternum).astype(np.int16), "vertebrae": count.copy()}
        finite = np.isfinite(sternum)
        total[finite] += sternum[finite]
        count += finite
        scores["integrated"], counts["integrated"] = mean_score(total, count), count
        for method, score in scores.items():
            true = score[own, mate[rows]]
            rank = np.where(np.isfinite(true), (score >= true[:, None]).sum(axis=1), np.nan)  # ties count against
            n_scored = np.isfinite(score).sum(axis=1)
            top = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1, kind="stable")[:, :2]  # ties: pid order
            top_score = score[own[:, None], top]
            impostor = score.copy()
            impostor[own, mate[rows]] = np.nan
            histogram[method] += np.histogram(impostor[np.isfinite(impostor)], bins=N_BINS, range=(0, 1))[0]
            if method == "integrated":
                impostor_idx[rows] = np.argmax(np.nan_to_num(impostor, nan=-np.inf), axis=1)
            records[method].append(pd.DataFrame({
                "pid": query_pids[rows], "method": method, "n_common": counts[method][own, mate[rows]],
                "n_scored": n_scored, "true_score": true, "true_rank": rank,
                "top1_pid": np.where(n_scored > 0, gallery_pids[top[:, 0]], ""),
                "top1_score": np.where(n_scored > 0, top_score[:, 0], np.nan),
                "top2_score": np.where(n_scored > 1, top_score[:, 1], np.nan)}))
    seconds = time.perf_counter() - started
    q_idx = np.arange(len(query_pids))
    structures = pd.DataFrame({"pid": query_pids, **{"mate_" + name: mate_similarity[name] for name in STRUCTURES}})
    structures["impostor_pid"] = gallery_pids[impostor_idx]  # integrated rank-1 among the other gallery entries
    for name in STRUCTURES:
        structures["impostor_" + name] = pair_similarity(query[name], gallery[name], q_idx, impostor_idx)
    structures = structures.assign(**shifted_vertebrae_scores(query_raw, gallery, scaling, mate))
    hist_rows = [{"method": method, "bin": int(b), "count": int(c)} for method, counts in histogram.items()
                 for b, c in zip(np.nonzero(counts)[0], counts[counts > 0], strict=True)]
    results = pd.concat([frame for method in METHODS for frame in records[method]], ignore_index=True)
    return {"matching_results": results[RESULT_FIELDS],
            "query_structures": structures,
            "impostor_histogram": pd.DataFrame(hist_rows, columns=["method", "bin", "count"]),
            "seconds": seconds, "n_gallery": len(gallery_pids), "n_query": len(query_pids)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="completed shape_features.csv")
    ap.add_argument("--qc", required=True, help="completed segmentation_qc.csv")
    ap.add_argument("--selection-dir", required=True, help="directory with exam_selection.csv")
    ap.add_argument("--nifti-root", required=True, help="directory with nifti_manifest.csv (ties the selection)")
    ap.add_argument("--seg-root", required=True, help="directory with segmentation_manifest.csv (ties the features)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    with output_lock(os.path.join(args.out_dir, "matching_results.csv")):
        run_matching(args)


def run_matching(args: argparse.Namespace) -> None:
    features_run = require_completion(args.features)
    require_completion(args.qc)
    selection_path = os.path.join(args.selection_dir, "exam_selection.csv")
    require_completion(selection_path)
    nifti_run = require_completion(os.path.join(args.nifti_root, "nifti_manifest.csv"))
    if nifti_run["run_identity"]["inputs"].get("exam_selection.csv") != file_hash(selection_path):
        raise ValueError("Selection differs from the one the conversion used")
    if features_run["run_identity"]["inputs"].get("qc") != file_hash(args.qc):
        raise ValueError("Feature table was made from a different QC table")
    seg_run = require_completion(os.path.join(args.seg_root, "segmentation_manifest.csv"))
    if features_run["run_identity"]["inputs"].get("segmentation_manifest.csv") != seg_run["sha256"]:
        raise ValueError("Feature table was made from a different segmentation manifest")
    if seg_run["run_identity"]["inputs"].get("nifti_manifest.csv") != nifti_run["sha256"]:
        raise ValueError("Segmentation was made from a different NIfTI manifest")
    if not {"T0", "T2"} <= set(features_run["run_identity"]["parameters"]["timepoints"].split(",")):
        raise ValueError("Feature extraction did not cover both T0 and T2")
    out = os.path.join(args.out_dir, "matching_results.csv")
    log = start_run(__file__, out, args, {"features": FEATURES, "methods": METHODS, "n_bins": N_BINS})
    read = dict(dtype=str, keep_default_na=False)
    qc = pd.read_csv(args.qc, **read)
    cohort = assign_roles(qc[qc["timepoint"].isin(["T0", "T2"])], pd.read_csv(selection_path, **read))
    features = pd.read_csv(args.features, **read)
    features = features[features["timepoint"].isin(["T0", "T2"])]  # exactly the usable structures (03 completion)
    roles = cohort["role"].value_counts().to_dict()
    log.info("eligible examinations: %d, roles: %s", len(cohort), roles)
    tables = match(features, cohort)
    log.info("matched %d queries against %d gallery examinations in %.1f s (%.2f ms per query)",
             tables["n_query"], tables["n_gallery"], tables["seconds"], 1000 * tables["seconds"] / tables["n_query"])
    files = {}
    for name in ("cohort", "query_structures", "impostor_histogram"):
        path = os.path.join(args.out_dir, name + ".csv")
        write_csv_atomic(cohort if name == "cohort" else tables[name], path)
        files[name + ".csv"] = file_hash(path)
    write_csv_atomic(tables["matching_results"], out)
    mark_completion(out, True, n_gallery=tables["n_gallery"], n_query=tables["n_query"],
                    n_no_reference=roles.get("no_reference", 0), n_single_structure=roles.get("single_structure", 0),
                    seconds_matching=round(tables["seconds"], 1),
                    ms_per_query=round(1000 * tables["seconds"] / tables["n_query"], 3), files=files)


if __name__ == "__main__":
    main()
