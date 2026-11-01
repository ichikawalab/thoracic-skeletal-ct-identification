"""Stage 1b: choose one validated series per examination (pid, time point) from the inventory.

Output (--out-dir): series_eligibility.csv (every series with its criteria), exam_selection.csv (one row per
examination with status selected/hold/incomplete/none and the chosen series), subject_timepoints.csv (per
subject: eligible time points, T0-T2 interval and acquisition agreement flags), selection_flowchart.csv and
kernel_summary.csv. Identity doubts (mixed patients or time points, shared UIDs) hold every examination they
could belong to. Eligible series (by default: valid, axial, thickness <= 3.2 mm, maximum spacing <= 1.01 x thickness)
are ranked by extent along the slice normal (ties within EXTENT_TIE_MM), then soft > sharp > unclassified kernel,
thinner slices, SeriesNumber, UID and path.
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd

from common.dicom_series import INVENTORY_FIELDS, ROUNDING_MM
from common.provenance import (
    TIME_POINTS,
    mark_completion,
    output_lock,
    require_completion,
    start_run,
    write_csv_atomic,
)

EXTENT_TIE_MM = 0.05
KERNEL_CLASS_RANK = {"soft": 0, "sharp": 1, "unclassified": 2}
INCOMPLETE_REASONS = {"no_files", "unreadable_DICOM", "pixel_decode_error", "empty_subject_or_study"}
CRITERIA = [
    ("ok_timepoint", "requested time point"),
    ("ok_identity", "examination identity resolved"),
    ("ok_validation", "all-slice CT validation passed"),
    ("ok_axial", "slice normal along z"),
    ("ok_thickness", "slice thickness within limit"),
    ("ok_spacing", "spacing <= thickness"),
]
EXAM_COLUMNS = [
    "pid", "timepoint", "series_relpath", "StudyInstanceUID", "SeriesInstanceUID", "SeriesNumber", "StudyDate",
    "manufacturer_group", "ManufacturerModelName", "ConvolutionKernel", "kernel_class", "validation_status",
    "thickness_mm", "spacing_mm", "extent_mm", "n_files", "n_unique_slices", "n_duplicates",
    "PixelSpacing", "KVP", "XRayTubeCurrent", "Exposure", "varied_tags",
    "flag_instance_mismatch", "flag_multi_study", "flag_kernel_unclassified", "flag_rescale_varies",
    "source_sha256", "timepoint_evidence",
]


def exam_hold_sources(df: pd.DataFrame, input_holds) -> dict:
    """(reason, source series) evidence per affected examination, propagated to every possible identity."""
    held = {key: {("empty_subject_or_study", "")} for key in input_holds}
    conflict = df["validation_status"].isin(["hold", "incomplete"])
    ambiguous = {}
    for tag in ("StudyInstanceUID", "SeriesInstanceUID"):
        owners = df.loc[df[tag].ne(""), [tag, "pid", "ClinicalTrialTimePointID"]].drop_duplicates()
        ambiguous[tag] = set(owners.loc[owners[tag].duplicated(), tag])
        conflict |= df[tag].isin(ambiguous[tag])
    study_timepoints = df.groupby(["pid", "study_folder"])["observed_timepoints"].agg(";".join)
    for row in df[conflict].itertuples():
        reasons = set(row.validation_reason.split(";")) - {""}
        if row.local_validation_status == "rejected":  # a technical rejection is context, not an identity reason
            reasons &= {"global_UID_conflict"}
        reasons |= {tag + "_owner_conflict" for tag, uids in ambiguous.items() if getattr(row, tag) in uids}
        observed = set(row.observed_timepoints.split(";"))
        timepoints = set(TIME_POINTS)  # unreadable files or an unresolved time point leave every time point possible
        if row.n_unread == "0" and observed <= timepoints:
            # An input defect of a series with a certain time point holds that time point only; an identity doubt
            # extends to every time point seen in the same study folder.
            timepoints = observed if reasons <= INCOMPLETE_REASONS else (
                timepoints & set(study_timepoints[row.pid, row.study_folder].split(";")))
        patients = {row.pid} | set(row.observed_patient_ids.split(";")) - {""}
        source = "/".join([row.pid, row.study_folder, row.series_folder])
        for key in itertools.product(patients, timepoints):
            held.setdefault(key, set()).update((reason, source) for reason in reasons)
    return held


def add_derived_columns(df: pd.DataFrame, held) -> pd.DataFrame:
    df = df.copy()
    df["timepoint"] = df["ClinicalTrialTimePointID"]
    df["manufacturer_group"] = df["Manufacturer"].str.upper().str.split().str[0].fillna("")
    df["series_relpath"] = df["pid"] + "/" + df["study_folder"] + "/" + df["series_folder"]
    for column, source in (("thickness_mm", "SliceThickness"), ("spacing_mm", "spacing_median_mm"),
                           ("extent_mm", "extent_mm"), ("normal_z", "normal_z")):
        df[column] = pd.to_numeric(df[source], errors="coerce")
    df["flag_instance_mismatch"] = df["flag_instance_mismatch"].str.lower().eq("true")
    df["flag_multi_study"] = df.groupby(["pid", "timepoint"])["StudyInstanceUID"].transform("nunique") > 1
    df["exam_hold"] = [key in held for key in zip(df["pid"], df["timepoint"], strict=True)]
    return df


def add_criteria(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df["ok_timepoint"] = df["timepoint"].isin(args.timepoints.split(","))
    df["ok_identity"] = ~df["exam_hold"]
    df["ok_validation"] = df["validation_status"].eq("valid")
    df["ok_axial"] = df["normal_z"] >= 1 - args.axial_tol
    df["ok_thickness"] = df["thickness_mm"].gt(0) & df["thickness_mm"].le(args.max_thickness)
    spacing_max = pd.to_numeric(df["spacing_max_mm"], errors="coerce")
    df["ok_spacing"] = df["spacing_mm"].gt(0) & spacing_max.le(
        df["thickness_mm"] * (1 + args.spacing_tol) + ROUNDING_MM)
    df["eligible"] = np.logical_and.reduce([df[column].to_numpy() for column, _ in CRITERIA])
    return df


def load_kernel_table(path: str) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not table["kernel_class"].isin(KERNEL_CLASS_RANK).all():
        raise ValueError("Unknown kernel class")
    return table.rename(columns={"manufacturer": "manufacturer_group", "kernel": "ConvolutionKernel"})


def add_kernel_class(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(table[["manufacturer_group", "ConvolutionKernel", "kernel_class"]],
                  on=["manufacturer_group", "ConvolutionKernel"], how="left", validate="many_to_one")
    df["kernel_class"] = df["kernel_class"].fillna("unclassified")
    df["flag_kernel_unclassified"] = df["kernel_class"].eq("unclassified")
    return df


def rank_within_exam(df: pd.DataFrame) -> pd.DataFrame:
    """Greatest extent first; extents within EXTENT_TIE_MM of their band's maximum tie (ties are not chained)."""
    order = df[df["eligible"]].sort_values(["pid", "timepoint", "extent_mm"], ascending=[True, True, False])
    bands, key, maximum = [], None, np.inf
    for pid, tp, extent in zip(order["pid"], order["timepoint"], order["extent_mm"], strict=True):
        if (pid, tp) != key or maximum - extent > EXTENT_TIE_MM + ROUNDING_MM:
            key, maximum = (pid, tp), extent
        bands.append(maximum)
    order = order.assign(extent_band=bands, class_rank=order["kernel_class"].map(KERNEL_CLASS_RANK),
                         series_number=pd.to_numeric(order["SeriesNumber"], errors="coerce").fillna(np.inf))
    order = order.sort_values(["pid", "timepoint", "extent_band", "class_rank", "thickness_mm", "series_number",
                               "SeriesInstanceUID", "series_relpath"], ascending=[True, True, False] + [True] * 5)
    ranks = order.groupby(["pid", "timepoint"]).cumcount() + 1
    df["rank_in_exam"] = ranks.reindex(df.index).astype("Int64")
    df["selected"] = df.index.isin(ranks.index[ranks == 1])
    return df


def exam_table(df: pd.DataFrame, timepoints, sources: dict) -> pd.DataFrame:
    requested = df[df["timepoint"].isin(timepoints)]
    keys = set(zip(requested["pid"], requested["timepoint"], strict=True)) | {k for k in sources if k[1] in timepoints}
    exams = pd.DataFrame(sorted(keys), columns=["pid", "timepoint"])
    counts = requested.groupby(["pid", "timepoint"], as_index=False).agg(
        n_series=("SeriesInstanceUID", "size"), n_eligible=("eligible", "sum"))
    exams = exams.merge(counts, how="left").fillna({"n_series": 0, "n_eligible": 0}).astype({"n_series": int,
                                                                                              "n_eligible": int})
    exams = exams.merge(df.loc[df["selected"], EXAM_COLUMNS], how="left", validate="one_to_one")
    evidence = [sources.get(key, set()) for key in zip(exams["pid"], exams["timepoint"], strict=True)]
    exams["status"] = ["hold" if {r for r, _ in e} - INCOMPLETE_REASONS else "incomplete" if e
                       else "selected" if n > 0 else "none"
                       for e, n in zip(evidence, exams["n_eligible"], strict=True)]
    exams["exam_eligible"] = exams["status"].eq("selected")
    exams["hold_reason"] = [";".join(sorted({r for r, _ in e})) for e in evidence]
    exams["hold_series"] = [";".join(sorted({s for _, s in e if s})) for e in evidence]
    return exams


def subject_table(exams: pd.DataFrame, subjects, empty_subjects) -> pd.DataFrame:
    index = pd.Index(sorted(subjects), name="pid")
    result = pd.DataFrame(index=index)
    result["n_timepoint_records"] = exams.groupby("pid").size().reindex(index, fill_value=0)
    result["n_exams"] = exams[exams["n_series"].gt(0)].groupby("pid").size().reindex(index, fill_value=0)
    for tp in TIME_POINTS:
        for flag, status in (("eligible", "selected"), ("hold", "hold"), ("incomplete", "incomplete")):
            affected = exams.loc[exams["timepoint"].eq(tp) & exams["status"].eq(status), "pid"]
            result[f"{flag}_{tp}"] = index.isin(affected)
    result["n_eligible_timepoints"] = result[[f"eligible_{tp}" for tp in TIME_POINTS]].sum(axis=1)
    result["has_T0_and_T2"] = result["eligible_T0"] & result["eligible_T2"]
    # Review aids for the selected T0/T2 pair; they do not change selection. The interval magnitude depends on
    # how StudyDate was de-identified, so only a T2 that is not after T0 is flagged.
    selected = exams[exams["exam_eligible"]].set_index(["pid", "timepoint"])

    def per_timepoint(column):
        return selected[column].unstack("timepoint").reindex(index=index, columns=["T0", "T2"])

    dates = per_timepoint("StudyDate")
    interval = (pd.to_datetime(dates["T2"], format="%Y%m%d", errors="coerce")
                - pd.to_datetime(dates["T0"], format="%Y%m%d", errors="coerce"))
    result["interval_days_T0_T2"] = interval.dt.days.astype("Int64")
    result["flag_T2_not_after_T0"] = result["interval_days_T0_T2"] <= 0
    for name, column in (("kernel_class", "kernel_class"), ("thickness", "thickness_mm"),
                         ("model", "ManufacturerModelName")):
        values = per_timepoint(column)
        result[f"same_{name}_T0_T2"] = (values["T0"] == values["T2"]).where(result["has_T0_and_T2"]).astype("boolean")
    result["input_status"] = np.where(index.isin(empty_subjects), "no_series", "inventoried")
    return result.reset_index()


def flowchart(df: pd.DataFrame, exams: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    remaining = pd.Series(True, index=df.index)
    rows = [("series", "inventoried", len(df), 0)]
    for column, description in CRITERIA:
        before = int(remaining.sum())
        remaining &= df[column]
        rows.append(("series", description, int(remaining.sum()), before - int(remaining.sum())))
    rows.append(("exam", "requested timepoint records (including holds)", len(exams), 0))
    rows += [("exam", status, int(exams["status"].eq(status).sum()), 0)
             for status in ("selected", "hold", "incomplete", "none")]
    rows += [("subject", "subjects inventoried", len(subjects), 0),
             ("subject", "eligible T0 and T2", int(subjects["has_T0_and_T2"].sum()), 0)]
    return pd.DataFrame(rows, columns=["level", "stage", "n_remaining", "n_excluded"])


def kernel_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Vendor/kernel counts over ORIGINAL/PRIMARY/AXIAL series and the three most frequent partner kernels."""
    keys = ["manufacturer_group", "ConvolutionKernel"]
    axial = df[df["ImageType"].map(lambda v: v.split("\\")[:3] == ["ORIGINAL", "PRIMARY", "AXIAL"]).astype(bool)]
    summary = axial.groupby(keys).agg(
        kernel_class=("kernel_class", "first"), n_series=("pid", "size"), n_subjects=("pid", "nunique"),
        thickness_median_mm=("thickness_mm", "median"),
        filter_types=("FilterType", lambda s: "; ".join(sorted(set(s)))))
    exams = axial.groupby(["pid", "timepoint", "manufacturer_group"])["ConvolutionKernel"].agg(lambda s: sorted(set(s)))
    pairs = pd.DataFrame([(group, a, b) for (_, _, group), kernels in exams.items()
                          for a in kernels for b in kernels if a != b], columns=[*keys, "partner"])
    counts = pairs.value_counts().reset_index(name="n")
    top = counts.sort_values([*keys, "n", "partner"], ascending=[True, True, False, True]).groupby(keys).head(3)
    partners = (top["partner"] + "(" + top["n"].astype(str) + ")").groupby([top[key] for key in keys]).agg("; ".join)
    summary["paired_with"] = partners.reindex(summary.index).fillna("")
    return summary.reset_index().sort_values(["manufacturer_group", "n_series"], ascending=[True, False],
                                             ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True, help="completed series_inventory.csv")
    ap.add_argument("--kernel-table", required=True, help="CSV with manufacturer, kernel and kernel_class")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--timepoints", default="T0,T2")
    ap.add_argument("--max-thickness", type=float, default=3.2, help="mm")
    ap.add_argument("--spacing-tol", type=float, default=0.01, help="relative tolerance for spacing <= thickness")
    ap.add_argument("--axial-tol", type=float, default=0.01, help="accept |normal_z| >= 1 - axial-tol")
    ap.add_argument("--clinical-screening", help="optional CSV with pid and study_yr (0, 1, 2) for corroboration")
    args = ap.parse_args()
    if not set(args.timepoints.split(",")) <= set(TIME_POINTS):
        ap.error("Unknown time point")
    if not 0 < args.max_thickness < np.inf or not 0 <= args.spacing_tol < 1 or not 0 <= args.axial_tol < 1:
        ap.error("Invalid geometry threshold")
    with output_lock(os.path.join(args.out_dir, "exam_selection.csv")):
        run_selection(args)


def run_selection(args: argparse.Namespace) -> None:
    completion = require_completion(args.inventory)
    selection_path = os.path.join(args.out_dir, "exam_selection.csv")
    log = start_run(__file__, selection_path, args)
    df = pd.read_csv(args.inventory, dtype=str, keep_default_na=False)
    if list(df.columns) != INVENTORY_FIELDS:
        raise ValueError("Inventory columns differ from this revision; rerun the inventory")
    df["timepoint_evidence"] = "header_consistent_only"
    if args.clinical_screening:
        clinical = pd.read_csv(args.clinical_screening, dtype=str, keep_default_na=False)
        if (not {"pid", "study_yr"} <= set(clinical.columns) or not clinical["study_yr"].isin(["0", "1", "2"]).all()
                or clinical["pid"].eq("").any() or clinical.duplicated(["pid", "study_yr"]).any()):
            raise ValueError("Clinical screening CSV needs unique, nonempty pid,study_yr rows with study_yr in 0,1,2")
        confirmed = set(zip(clinical["pid"], "T" + clinical["study_yr"], strict=True))
        df["timepoint_evidence"] = ["clinical_row_matched" if key in confirmed else "clinical_row_unavailable"
                                    for key in zip(df["pid"], df["ClinicalTrialTimePointID"], strict=True)]
    unresolved = ~df["observed_timepoints"].isin(TIME_POINTS) | df["n_unread"].ne("0")
    df.loc[unresolved, "timepoint_evidence"] = "header_unresolved"
    missing = set(completion["empty_subjects"]) | {pid for pid, _ in completion["empty_studies"]}
    sources = exam_hold_sources(df, {(pid, tp) for pid in missing for tp in TIME_POINTS})
    df = add_criteria(add_derived_columns(df, set(sources)), args)
    df = rank_within_exam(add_kernel_class(df, load_kernel_table(args.kernel_table)))
    exams = exam_table(df, args.timepoints.split(","), sources)
    subjects = subject_table(exams, completion["subjects"], completion["empty_subjects"])
    chart = flowchart(df, exams, subjects)
    kernels = kernel_summary(df)
    for name, table in (("series_eligibility", df), ("exam_selection", exams), ("subject_timepoints", subjects),
                        ("selection_flowchart", chart), ("kernel_summary", kernels)):
        write_csv_atomic(table, os.path.join(args.out_dir, name + ".csv"))
    mark_completion(selection_path, True, inventory_sha256=completion["sha256"])
    for row in kernels[kernels["kernel_class"].eq("unclassified")].itertuples():
        log.info("unclassified kernel: %s %s (%d series)", row.manufacturer_group, row.ConvolutionKernel, row.n_series)
    for tp, extent in exams[exams["exam_eligible"]].groupby("timepoint")["extent_mm"]:
        log.info("%s selected scan extent (mm): min=%.2f p01=%.2f median=%.2f max=%.2f",
                 tp, extent.min(), extent.quantile(0.01), extent.median(), extent.max())
    for row in chart.itertuples(index=False):
        log.info("%-8s %-42s n=%d", row.level, row.stage, row.n_remaining)


if __name__ == "__main__":
    main()
