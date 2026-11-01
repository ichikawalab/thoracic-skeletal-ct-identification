"""Examination and structure rules of the segmentation QC on synthetic masks, and its per-examination resume."""
import json

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from tests.conftest import load_stage_script

qc = load_stage_script("02_segmentation_qc/qc_segmentation.py")
LABELS = {"sternum": 116, **{f"vertebrae_T{i}": 7 + i for i in range(1, 13)}}
EXAM = {"pid": "1", "timepoint": "T0", "sternum_path": "sternum.nii.gz", "vertebrae_path": "vertebrae.nii.gz"}


def write_masks(directory, sternum_y=slice(3, 6), vertebrae_y=slice(14, 17), sternum_z=slice(4, 8), flip=False,
                absent=12):
    """LPS masks [z, y, x]: sternum at low y (anterior), T1-T12 at high y; T1 on the first slice, ``absent`` missing."""
    sternum = np.zeros((16, 20, 20), dtype=np.uint8)
    sternum[sternum_z, sternum_y, 8:12] = LABELS["sternum"]
    vertebrae = np.zeros_like(sternum)
    vertebrae[0:2, vertebrae_y, 8:12] = LABELS["vertebrae_T1"]
    for i in range(2, 13):
        if i != absent:
            vertebrae[i, vertebrae_y, 8:12] = LABELS[f"vertebrae_T{i}"]
    for name, array in (("sternum", sternum), ("vertebrae", vertebrae)):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((0.7, 0.8, 2.5))
        if flip:  # the same physical scene stored with reversed y and z axes
            image = sitk.Flip(image, [False, True, True])
        sitk.WriteImage(image, str(directory / f"{name}.nii.gz"))


def states(rows):
    return {row["structure"]: row["structure_status"] for row in rows}


@pytest.mark.parametrize("flip", [False, True])
def test_eligible_exam_reports_structure_states_independent_of_stored_orientation(tmp_path, flip):
    write_masks(tmp_path, flip=flip)
    rows = qc.qc_exam(str(tmp_path), EXAM, LABELS)
    assert len(rows) == 13 and all(row["status"] == "complete" and row["error"] == "" for row in rows)
    first = rows[0]
    assert first["exam_eligible"] is True and first["exam_exclusion"] == ""
    assert first["sternum_present"] and first["thoracic_present"] and first["sternum_within_fov"]
    assert first["sternum_anterior_mm"] == pytest.approx(11 * 0.8, abs=0.05)  # 11 voxel rows of 0.8 mm
    expected = {name: "usable" for name in qc.STRUCTURES}
    expected.update(vertebrae_T1="truncated", vertebrae_T12="absent")
    assert states(rows) == expected and first["vertebra_numbering_gap"] is False
    t1 = next(row for row in rows if row["structure"] == "vertebrae_T1")
    assert t1["n_voxels"] == 2 * 3 * 4 and t1["touches_z_boundary"] and not t1["touches_xy_boundary"]


def test_sternum_behind_the_vertebrae_makes_the_exam_ineligible(tmp_path):
    write_masks(tmp_path, sternum_y=slice(14, 17), vertebrae_y=slice(3, 6))
    rows = qc.qc_exam(str(tmp_path), EXAM, LABELS)
    assert rows[0]["exam_exclusion"] == "sternum_anterior" and rows[0]["sternum_anterior_mm"] < 0
    assert set(states(rows).values()) == {"ineligible_exam"}
    assert next(row for row in rows if row["structure"] == "vertebrae_T12")["n_voxels"] == 0


def test_missing_vertebra_between_present_ones_disables_every_vertebra_but_not_the_sternum(tmp_path):
    write_masks(tmp_path, absent=6)
    rows = qc.qc_exam(str(tmp_path), EXAM, LABELS)
    assert rows[0]["exam_eligible"] is True and rows[0]["vertebra_numbering_gap"] is True
    assert states(rows) == {"sternum": "usable", **dict.fromkeys(qc.THORACIC, "numbering_gap")}
    assert next(row for row in rows if row["structure"] == "vertebrae_T1")["touches_z_boundary"]  # flags are kept


def test_missing_or_truncated_sternum_is_an_exam_exclusion(tmp_path):
    write_masks(tmp_path, sternum_y=slice(0, 0))
    rows = qc.qc_exam(str(tmp_path), EXAM, LABELS)
    assert rows[0]["exam_exclusion"] == "sternum_present" and rows[0]["sternum_anterior_mm"] == ""
    write_masks(tmp_path, sternum_z=slice(12, 16))
    rows = qc.qc_exam(str(tmp_path), EXAM, LABELS)
    assert rows[0]["exam_exclusion"] == "sternum_within_fov" and rows[0]["sternum_anterior_mm"] > 0
    assert states(rows)["sternum"] == "ineligible_exam"


def test_unreadable_mask_is_a_read_error_for_every_structure(tmp_path):
    write_masks(tmp_path)
    rows = qc.qc_exam(str(tmp_path), {**EXAM, "vertebrae_path": "missing.nii.gz"}, LABELS)
    assert len(rows) == 13 and all(row["status"] == "error" and row["error"].startswith("read_error") for row in rows)


def test_label_numbers_come_from_the_task_that_produced_each_mask(tmp_path):
    maps = {"total": {"116": "sternum", "32": "vertebrae_T12"},
            "vertebrae_pp_refined": {str(43 - i): f"vertebrae_T{i + 1}" for i in range(12)} | {"31": "vertebrae_L1"}}
    (tmp_path / "label_maps.json").write_text(json.dumps(maps))
    labels = qc.label_numbers(str(tmp_path))
    assert labels["sternum"] == 116 and labels["vertebrae_T1"] == 43 and labels["vertebrae_T12"] == 32
    assert set(labels) == set(qc.STRUCTURES)
    del maps["vertebrae_pp_refined"]["32"]
    (tmp_path / "label_maps.json").write_text(json.dumps(maps))
    with pytest.raises(KeyError, match="vertebrae_T12"):
        qc.label_numbers(str(tmp_path))


def qc_rows(pid="1", error=""):
    rows = [dict.fromkeys(qc.FIELDS, "") for _ in qc.STRUCTURES]
    for row, name in zip(rows, qc.STRUCTURES, strict=True):
        row.update(pid=pid, timepoint="T0", structure=name, structure_status="usable", error=error,
                   status="error" if error else "complete")
    return rows


def test_resume_needs_all_thirteen_records_of_an_examination(tmp_path):
    path = tmp_path / "qc.csv"
    pd.DataFrame(qc_rows() + qc_rows("2")[:12] + qc_rows("3", "read_error: temporary")).to_csv(path, index=False)
    assert qc.resume_qc(str(path), {("1", "T0"), ("2", "T0"), ("3", "T0")}) == {("1", "T0")}
    pd.DataFrame(qc_rows("3")).to_csv(path, mode="a", header=False, index=False)
    assert qc.resume_qc(str(path), {("1", "T0"), ("2", "T0"), ("3", "T0")}) == {("1", "T0"), ("3", "T0")}
    assert len(pd.read_csv(path)) == 38  # compacted to the latest record per structure
    with pytest.raises(ValueError, match="differs"):
        qc.resume_qc(str(path), {("1", "T0")})
