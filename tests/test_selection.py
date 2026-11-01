"""Selection ranks inspected series and preserves examination-level identity and input holds."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.dicom_series import AUDIT_FIELDS, SUMMARY_TAGS
from tests.conftest import load_stage_script, make_ct_series

select = load_stage_script("01_preprocessing/select_series.py")
inventory = load_stage_script("01_preprocessing/inventory_dicom_series.py")


def series(pid, tp, uid, **overrides):
    row = dict.fromkeys([*SUMMARY_TAGS, *AUDIT_FIELDS], "")
    row.update(pid=pid, study_folder="s" + tp, series_folder=uid, n_files="120", n_unread="0",
               PatientID=pid, ClinicalTrialTimePointID=tp, StudyInstanceUID="study-" + pid + tp,
               SeriesInstanceUID=uid, SeriesNumber="2", Manufacturer="GE", ConvolutionKernel="STANDARD",
               ImageType="ORIGINAL\\PRIMARY\\AXIAL", SliceThickness="2.5", spacing_median_mm="2.5",
               spacing_max_mm="2.5", extent_mm="297.5", normal_z="1", n_unique_slices="120", n_duplicates="0",
               flag_instance_mismatch="False", validation_status="valid", observed_timepoints=tp,
               observed_patient_ids=pid, timepoint_evidence="header_consistent_only")
    row.update(overrides)
    row.setdefault("local_validation_status", row["validation_status"])
    return row


@pytest.fixture
def args(tmp_path):
    table = tmp_path / "kernels.csv"
    table.write_text("manufacturer,kernel,kernel_class\nGE,STANDARD,soft\nGE,BONE,sharp\n")
    return argparse.Namespace(max_thickness=3.2, spacing_tol=0.01, axial_tol=0.01,
                              timepoints="T0,T2", kernel_table=str(table))


def run(rows, args, input_holds=()):
    df = pd.DataFrame(rows)
    df = select.add_criteria(select.add_derived_columns(df, set(select.exam_hold_sources(df, input_holds))), args)
    return select.rank_within_exam(select.add_kernel_class(df, select.load_kernel_table(args.kernel_table)))


def exams_of(df, input_holds=()):
    return select.exam_table(df, ("T0", "T2"), select.exam_hold_sources(df, input_holds))


@pytest.mark.parametrize("overrides,criterion", [
    (dict(validation_status="rejected"), "ok_validation"),
    (dict(SliceThickness="5"), "ok_thickness"),
    (dict(spacing_max_mm="5"), "ok_spacing"),
    (dict(normal_z="0.9"), "ok_axial"),
    (dict(ClinicalTrialTimePointID="T1"), "ok_timepoint"),
])
def test_selection_criteria(overrides, criterion, args):
    df = run([series("1", "T0", "a", **overrides)], args)
    assert not df.iloc[0][criterion] and not df.iloc[0]["selected"]


def test_gaps_are_checked_after_deduplication(args):
    df = run([series("1", "T0", "a", n_files="240", n_duplicates="120",
                     spacing_median_mm="5", spacing_max_mm="5", extent_mm="595")], args)
    assert not df.iloc[0]["ok_spacing"]


def test_overlapping_slices_are_allowed(args):
    df = run([series("1", "T0", "a", spacing_median_mm="1.25", spacing_max_mm="1.25")], args)
    assert df.iloc[0]["selected"]


@pytest.mark.parametrize("extent", ["2.5", "50", "150", "200"])
def test_scan_extent_does_not_exclude_series(args, extent):
    assert run([series("1", "T0", "a", extent_mm=extent)], args).iloc[0]["selected"]


def test_ranking_and_next_valid_series(args):
    rows = [
        series("1", "T0", "broken", validation_status="rejected", SliceThickness="1.25"),
        series("1", "T0", "soft"),
        series("1", "T0", "soft-thin", SliceThickness="1.25", spacing_max_mm="1.25", n_unique_slices="200"),
        series("1", "T0", "soft-mismatch", flag_instance_mismatch="True", SliceThickness="1", spacing_max_mm="1"),
        series("1", "T0", "sharp", ConvolutionKernel="BONE"),
        series("1", "T0", "unknown", ConvolutionKernel="EDGE"),
    ]
    df = run(rows, args).set_index("SeriesInstanceUID")
    ranks = df["rank_in_exam"]
    assert ranks[["soft-mismatch", "soft-thin", "soft", "sharp", "unknown"]].tolist() == [1, 2, 3, 4, 5]
    assert pd.isna(ranks["broken"]) and df["rank_in_exam"].dtype == "Int64"


def test_full_coverage_precedes_thinner_short_series(args):
    df = run([series("1", "T0", "short", extent_mm="1", SliceThickness="1", spacing_max_mm="1"),
              series("1", "T0", "full", extent_mm="291")], args)
    assert df.loc[df.selected, "SeriesInstanceUID"].tolist() == ["full"]


def test_extent_ties_are_not_chained(args):
    df = run([series("1", "T0", "max", extent_mm="300", ConvolutionKernel="BONE"),
              series("1", "T0", "tied", extent_mm="299.96"),
              series("1", "T0", "outside", extent_mm="299.92", SliceThickness="1", spacing_max_mm="1")], args)
    assert df.loc[df.selected, "SeriesInstanceUID"].tolist() == ["tied"]
    assert df.set_index("SeriesInstanceUID").loc["outside", "rank_in_exam"] == 3


@pytest.mark.parametrize("shorter,selected", [(299.95 + 1e-8, "soft"), (299.95, "soft"), (299.95 - 1e-8, "sharp")])
def test_extent_tie_includes_the_boundary(args, shorter, selected):
    df = run([series("1", "T0", "sharp", extent_mm="300", ConvolutionKernel="BONE"),
              series("1", "T0", "soft", extent_mm=str(shorter))], args)
    assert df.loc[df.selected, "SeriesInstanceUID"].tolist() == [selected]


@pytest.mark.parametrize("excess,selected", [(-1e-8, True), (0, True), (1e-8, False)])
def test_spacing_limit_includes_the_boundary(args, excess, selected):
    spacing = (100 + 2.525) - 100 + excess  # 2.5 mm thickness with 1% spacing tolerance
    df = run([series("1", "T0", "a", spacing_median_mm=str(spacing), spacing_max_mm=str(spacing))], args)
    assert bool(df.iloc[0].selected) == selected


@pytest.mark.parametrize("representation,word", [(1, 0xFC18), (1, 0x0C18), (1, 0xAC18), (0, 0x13E8)])
def test_unused_pixel_bits_do_not_exclude_a_series(tmp_path, args, representation, word):
    directory = tmp_path / "ct"
    for ds in make_ct_series(directory):
        ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 12, 11, representation
        ds.PixelData = np.full((6, 8), word, dtype=np.uint16).tobytes()
        ds.save_as(ds.filename, write_like_original=False)
    row, _ = inventory.scan_series("1", "study", str(directory))
    df = run([{key: str(value) for key, value in row.items()}], args)
    assert df.iloc[0].validation_status == "valid" and df.iloc[0].selected


def test_any_identity_conflict_holds_exam_but_not_other_timepoints(args):
    df = run([series("1", "T0", "baseline"), series("1", "T2", "good"),
              series("1", "T2", "conflict", validation_status="hold", observed_patient_ids="1;2",
                     validation_reason="patient_or_timepoint_conflict;global_UID_conflict")], args)
    exams = exams_of(df).set_index(["pid", "timepoint"])
    assert exams["status"].to_dict() == {("1", "T0"): "selected", ("1", "T2"): "hold", ("2", "T2"): "hold"}
    assert df.loc[df.timepoint.eq("T2"), "selected"].sum() == 0
    for pid in ("1", "2"):
        assert exams.loc[(pid, "T2"), "hold_reason"] == "global_UID_conflict;patient_or_timepoint_conflict"
        assert exams.loc[(pid, "T2"), "hold_series"] == "1/sT2/conflict"
    assert exams.loc[("1", "T0"), "hold_reason"] == exams.loc[("1", "T0"), "hold_series"] == ""


def test_mixed_timepoints_hold_both_possible_exams(args):
    df = run([series("1", "T0", "a", validation_status="hold", observed_timepoints="T0;T2",
                     validation_reason="patient_or_timepoint_conflict")], args)
    assert exams_of(df).set_index("timepoint")["status"].to_dict() == {"T0": "hold", "T2": "hold"}


def test_missing_timepoint_is_not_inferred_from_other_series(args):
    df = run([series("1", "", "bad", study_folder="sT2", validation_status="hold", observed_timepoints="",
                     validation_reason="patient_or_timepoint_conflict"),
              series("1", "T2", "good"), series("1", "T0", "baseline")], args)
    exams = exams_of(df).set_index("timepoint")
    assert exams["status"].to_dict() == {"T0": "hold", "T2": "hold"}
    assert exams["hold_reason"].eq("patient_or_timepoint_conflict").all()
    assert exams["hold_series"].eq("1/sT2/bad").all()


@pytest.mark.parametrize("tag", ["StudyInstanceUID", "SeriesInstanceUID"])
def test_uid_reuse_across_subjects_holds_both_exams(args, tag):
    df = run([series("1", "T0", "a", **{tag: "shared"}), series("2", "T0", "b", **{tag: "shared"})], args)
    exams = exams_of(df)
    assert exams["status"].tolist() == ["hold", "hold"]
    assert exams["hold_reason"].eq(tag + "_owner_conflict").all()
    assert exams["hold_series"].str.len().gt(0).all()


@pytest.mark.parametrize("tag", ["StudyInstanceUID", "SeriesInstanceUID"])
def test_uid_reuse_across_timepoints_holds_both_exams(args, tag):
    df = run([series("1", "T0", "a", **{tag: "shared"}), series("1", "T2", "b", **{tag: "shared"})], args)
    assert exams_of(df)["status"].tolist() == ["hold", "hold"]


def test_global_hold_preserves_local_rejection_without_making_it_a_hold_reason(args):
    df = run([series("1", "T0", "localizer", validation_status="hold", local_validation_status="rejected",
                     validation_reason="not_original_primary_axial;global_UID_conflict"),
              series("1", "T0", "good")], args)
    exam = exams_of(df).iloc[0]
    assert exam.status == "hold" and not df.selected.any()
    assert exam.hold_reason == "global_UID_conflict" and exam.hold_series == "1/sT0/localizer"
    assert df.iloc[0].validation_reason == "not_original_primary_axial;global_UID_conflict"


def test_incomplete_input_blocks_alternative_and_has_distinct_status(args):
    df = run([series("1", "T0", "broken", validation_status="incomplete", validation_reason="pixel_decode_error"),
              series("1", "T0", "good"), series("1", "T2", "followup")], args)
    exams = exams_of(df).set_index("timepoint")
    assert exams["status"].to_dict() == {"T0": "incomplete", "T2": "selected"}
    assert exams.loc["T0", ["hold_reason", "hold_series"]].tolist() == ["pixel_decode_error", "1/sT0/broken"]
    assert exams.loc["T2", "hold_reason"] == exams.loc["T2", "hold_series"] == ""


def test_unreadable_files_leave_every_time_point_possible(args):
    df = run([series("1", "T0", "broken", validation_status="incomplete", validation_reason="unreadable_DICOM",
                     n_unread="2"), series("1", "T2", "good")], args)
    exams = exams_of(df).set_index("timepoint")
    assert exams["status"].to_dict() == {"T0": "incomplete", "T2": "incomplete"}
    assert exams["hold_series"].eq("1/sT0/broken").all()


def test_input_defect_of_a_certain_time_point_does_not_hold_the_other(args):
    df = run([series("1", "T0", "baseline", study_folder="visit"),
              series("1", "T2", "broken", study_folder="visit", validation_status="incomplete",
                     validation_reason="pixel_decode_error")], args)
    assert exams_of(df).set_index("timepoint")["status"].to_dict() == {"T0": "selected", "T2": "incomplete"}


def test_empty_study_does_not_hide_behind_another_candidate(args):
    holds = {("1", "T0"), ("1", "T2")}
    df = run([series("1", "T0", "candidate")], args, holds)
    assert not df.selected.any()
    assert exams_of(df, holds)["status"].eq("incomplete").all()


def test_manufacturer_group_is_normalized_and_columns_stay_distinct(args):
    df = run([series("1", "T0", "a", Manufacturer="GE MEDICAL SYSTEMS"), series("2", "T0", "b", Manufacturer="Philips"),
              series("3", "T0", "c", Manufacturer="")], args)
    assert df["manufacturer_group"].tolist() == ["GE", "PHILIPS", ""]
    assert df.iloc[0].Manufacturer == "GE MEDICAL SYSTEMS" and df.iloc[0].kernel_class == "soft"
    for table in (df, exams_of(df)):
        assert len({name.casefold() for name in table.columns}) == len(table.columns)


def test_subject_denominators_include_unassignable_subjects(args):
    df = run([series("1", "T0", "a"), series("1", "T2", "b"), series("2", "T0", "c"), series("2", "T1", "d"),
              series("3", "T2", "e"), series("4", "", "unread", validation_status="rejected")], args)
    exams = exams_of(df)
    subjects = select.subject_table(exams, df.pid.unique(), []).set_index("pid")
    assert subjects.loc["4", "n_exams"] == 0
    assert subjects.loc["2", "eligible_T0"] and subjects.loc["3", "eligible_T2"]
    chart = select.flowchart(df, exams, subjects.reset_index()).set_index("stage")
    assert chart.loc["subjects inventoried", "n_remaining"] == 4
    assert chart.loc["eligible T0 and T2", "n_remaining"] == 1


def test_subject_pair_flags_record_interval_and_acquisition_consistency(args):
    df = run([series("1", "T0", "a", StudyDate="20000105"),
              series("1", "T2", "b", StudyDate="20020105", SliceThickness="1.25", spacing_max_mm="1.25",
                     ConvolutionKernel="BONE"),
              series("2", "T0", "c", StudyDate="20010101"), series("2", "T2", "d", StudyDate="20001231"),
              series("3", "T0", "e")], args)
    subjects = select.subject_table(exams_of(df), df.pid.unique(), []).set_index("pid")
    assert subjects.loc["1", "interval_days_T0_T2"] == 731 and not subjects.loc["1", "flag_T2_not_after_T0"]
    assert subjects.loc["2", "interval_days_T0_T2"] == -1 and subjects.loc["2", "flag_T2_not_after_T0"]
    assert pd.isna(subjects.loc["3", "interval_days_T0_T2"]) and pd.isna(subjects.loc["3", "same_kernel_class_T0_T2"])
    same = ["same_kernel_class_T0_T2", "same_thickness_T0_T2", "same_model_T0_T2"]
    assert subjects.loc["1", same].tolist() == [False, False, True]
    assert subjects.loc["2", same].tolist() == [True, True, True]


def test_kernel_summary_counts_partners_within_manufacturer(args):
    rows = [series("1", "T0", "a"), series("1", "T0", "b", ConvolutionKernel="BONE"),
            series("2", "T0", "c", Manufacturer="OTHER"), series("2", "T0", "d", Manufacturer="OTHER",
                                                                 ConvolutionKernel="SHARP"),
            series("3", "T0", "e", ImageType="ORIGINAL\\PRIMARY\\LOCALIZER")]
    summary = select.kernel_summary(run(rows, args)).set_index(["manufacturer_group", "ConvolutionKernel"])
    columns = ["kernel_class", "n_series", "paired_with"]
    assert summary.loc[("GE", "STANDARD"), columns].tolist() == ["soft", 1, "BONE(1)"]
    assert summary.loc[("OTHER", "STANDARD"), columns].tolist() == ["unclassified", 1, "SHARP(1)"]
    assert len(summary) == 4
    assert select.kernel_summary(run(rows[:1], args))["paired_with"].tolist() == [""]


def test_selection_refuses_partial_inventory(tmp_path, monkeypatch, args):
    path = tmp_path / "inventory.csv"
    pd.DataFrame([series("1", "T0", "a")]).to_csv(path, index=False)
    monkeypatch.setattr(sys, "argv", ["select_series.py", "--inventory", str(path),
                        "--kernel-table", args.kernel_table, "--out-dir", str(tmp_path / "selection")])
    with pytest.raises(ValueError, match="Incomplete stage"):
        select.main()


def test_empty_subjects_and_studies_remain_input_incomplete(tmp_path, args):
    root = tmp_path / "dicom"
    make_ct_series(root / "1" / "study" / "series")
    (root / "2").mkdir()
    (root / "3" / "empty_study").mkdir(parents=True)
    inventory.run_inventory(argparse.Namespace(root=str(root), out=str(tmp_path / "inventory.csv"), workers=1, limit=0))
    args.inventory, args.out_dir, args.clinical_screening = str(tmp_path / "inventory.csv"), str(tmp_path / "sel"), None
    select.run_selection(args)
    subjects = pd.read_csv(Path(args.out_dir) / "subject_timepoints.csv", dtype=str)
    exams = pd.read_csv(Path(args.out_dir) / "exam_selection.csv", dtype=str, keep_default_na=False)
    assert subjects["n_exams"].tolist() == ["1", "0", "0"]
    assert subjects["n_timepoint_records"].tolist() == ["1", "2", "2"]
    assert subjects["input_status"].tolist() == ["inventoried", "no_series", "no_series"]
    held = exams[exams["pid"].ne("1")]
    assert len(held) == 4 and held["status"].eq("incomplete").all() and held["n_series"].eq("0").all()
    assert held["hold_reason"].eq("empty_subject_or_study").all() and held["hold_series"].eq("").all()
    assert exams.loc[exams["pid"].eq("1"), ["status", "n_series"]].values.tolist() == [["selected", "1"]]


def test_selection_runs_on_an_inventory_without_series(tmp_path, args):
    root = tmp_path / "dicom"
    (root / "1" / "empty_study").mkdir(parents=True)
    (root / "2").mkdir()
    inventory.run_inventory(argparse.Namespace(root=str(root), out=str(tmp_path / "inventory.csv"), workers=1, limit=0))
    args.inventory, args.out_dir, args.clinical_screening = str(tmp_path / "inventory.csv"), str(tmp_path / "sel"), None
    select.run_selection(args)
    exams = pd.read_csv(Path(args.out_dir) / "exam_selection.csv", dtype=str, keep_default_na=False)
    assert len(exams) == 4 and exams["status"].eq("incomplete").all()
    assert pd.read_csv(Path(args.out_dir) / "kernel_summary.csv").empty
    assert pd.read_csv(Path(args.out_dir) / "subject_timepoints.csv")["input_status"].eq("no_series").all()
