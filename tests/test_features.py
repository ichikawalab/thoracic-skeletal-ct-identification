"""Mask orientation handling, error propagation and per-structure resume of feature extraction."""
import argparse
import json

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from common.provenance import SEGMENTATION_FIELDS, file_hash, mark_completion, start_run
from tests.conftest import load_stage_script

extract = load_stage_script("03_feature_extraction/extract_shape_features.py")
STRUCTURES = ["sternum", *(f"vertebrae_T{i}" for i in range(1, 13))]
EXPECTED = {("1", "T0", name) for name in STRUCTURES}


def test_masks_with_permuted_axes_are_processed_in_the_same_orientation(tmp_path, monkeypatch):
    array = np.zeros((7, 8, 9), dtype=np.uint8)
    array[1:6, 2:6, 2:7] = 1
    original = sitk.GetImageFromArray(array)
    original.SetSpacing((0.7, 0.8, 2.5))
    sitk.WriteImage(original, str(tmp_path / "original.nii.gz"))
    sitk.WriteImage(sitk.PermuteAxes(original, [2, 0, 1]), str(tmp_path / "permuted.nii.gz"))
    monkeypatch.setattr(extract, "structure_row", lambda pid, tp, name, label, image, params:
                        (name, label, image.GetDirection(), image.GetSize(), sitk.GetArrayFromImage(image).tolist()))

    def run(path):
        exam = dict(pid="1", timepoint="T0", sternum_path=path, vertebrae_path=path,
                    structures=dict.fromkeys(STRUCTURES, 1))
        return extract.extract_exam(str(tmp_path), exam, "unused")

    assert run("original.nii.gz") == run("permuted.nii.gz")
    assert [row[0] for row in run("original.nii.gz")] == STRUCTURES


def test_unreadable_mask_fails_only_the_structures_it_serves(tmp_path, monkeypatch):
    sitk.WriteImage(sitk.GetImageFromArray(np.ones((3, 3, 3), dtype=np.uint8)), str(tmp_path / "sternum.nii.gz"))
    monkeypatch.setattr(extract, "structure_row", lambda pid, tp, name, label, image, params: {"structure": name})
    exam = dict(pid="1", timepoint="T0", sternum_path="sternum.nii.gz", vertebrae_path="missing.nii.gz",
                structures={"sternum": 1, "vertebrae_T5": 12})
    sternum, t5 = extract.extract_exam(str(tmp_path), exam, "unused")
    assert sternum == {"structure": "sternum"}
    assert (t5["structure"], t5["label"], t5["status"]) == ("vertebrae_T5", 12, "error")
    assert t5["error"].startswith("read_error")


def test_program_errors_are_not_reported_as_structure_errors(monkeypatch):
    def broken_extractor(params):
        raise TypeError("program defect")

    monkeypatch.setattr(extract, "get_extractor", broken_extractor)
    image = sitk.GetImageFromArray(np.ones((3, 3, 3), dtype=np.uint8))
    with pytest.raises(TypeError, match="program defect"):
        extract.structure_row("1", "T0", "sternum", 1, image, "unused")


def feature_rows(error=""):
    rows = [dict.fromkeys(extract.FIELDS, "") for _ in STRUCTURES]
    for row, name in zip(rows, STRUCTURES, strict=True):
        row.update(pid="1", timepoint="T0", structure=name, status="computed")
        row.update({feature: 1.5 for feature in extract.SHAPE_FEATURES})
        if name == "sternum" and error:
            row.update({feature: "" for feature in extract.SHAPE_FEATURES}, error=error, status="error")
    return rows


def test_retry_preserves_other_completed_structures(tmp_path):
    path = tmp_path / "features.csv"
    pd.DataFrame(feature_rows("read_error: temporary")).to_csv(path, index=False)
    assert extract.resume_features(str(path), EXPECTED) == EXPECTED - {("1", "T0", "sternum")}
    assert len(pd.read_csv(path)) == 13 and pd.read_csv(path).iloc[0]["error"] == "read_error: temporary"
    pd.DataFrame(feature_rows()[:1]).to_csv(path, mode="a", header=False, index=False)
    assert extract.resume_features(str(path), EXPECTED) == EXPECTED
    assert len(pd.read_csv(path)) == 13


def test_checkpoint_must_match_the_usable_structures(tmp_path):
    path = tmp_path / "features.csv"
    pd.DataFrame(feature_rows()[:1]).to_csv(path, index=False)
    assert extract.resume_features(str(path), EXPECTED) == {("1", "T0", "sternum")}
    with pytest.raises(ValueError, match="differs"):
        extract.resume_features(str(path), {("2", "T0", "sternum")})


def test_truncated_or_nonfinite_features_are_retried(tmp_path):
    path = tmp_path / "features.csv"
    row = feature_rows()[0]
    for field in ("status", extract.SHAPE_FEATURES[-1]):
        pd.DataFrame([{**row, field: ""}]).to_csv(path, index=False)
        assert extract.resume_features(str(path), EXPECTED) == set()


def test_retry_history_uses_latest_completed_values(tmp_path):
    path = tmp_path / "features.csv"
    pd.DataFrame(feature_rows("read_error: temporary") + feature_rows()).to_csv(path, index=False)
    assert len(extract.resume_features(str(path), EXPECTED)) == 13
    assert len(pd.read_csv(path)) == 13


def completed_inputs(tmp_path, qc_timepoints=("T0", "T2"), qc_seg_hash=None):
    """A completed two-examination segmentation root and a completed QC table made from it (or from another)."""
    seg_root = tmp_path / "seg"
    seg_root.mkdir()
    (seg_root / "label_maps.json").write_text("{}")
    rows = []
    for tp in ("T0", "T2"):
        for name in ("sternum", "vertebrae"):
            (seg_root / f"1_{tp}_{name}.nii.gz").write_bytes(tp.encode() + name.encode())
        rows.append({**dict.fromkeys(SEGMENTATION_FIELDS, ""), "pid": "1", "timepoint": tp,
                     "sternum_path": f"1_{tp}_sternum.nii.gz", "vertebrae_path": f"1_{tp}_vertebrae.nii.gz",
                     "sternum_sha256": file_hash(seg_root / f"1_{tp}_sternum.nii.gz"),
                     "vertebrae_sha256": file_hash(seg_root / f"1_{tp}_vertebrae.nii.gz"), "status": "complete"})
    manifest = seg_root / "segmentation_manifest.csv"
    start_run("run_segmentation.py", str(manifest), argparse.Namespace(out_root=str(seg_root)))
    pd.DataFrame(rows).to_csv(manifest, index=False)
    mark_completion(manifest, True)
    qc_path = tmp_path / "qc.csv"
    qc_rows = [{"pid": "1", "timepoint": tp, "structure": name, "label": 1, "structure_status": "usable"}
               for tp in qc_timepoints for name in STRUCTURES]
    start_run("qc_segmentation.py", str(qc_path), argparse.Namespace(seg_root=str(seg_root), out=str(qc_path)))
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
    mark_completion(qc_path, True)
    if qc_seg_hash:
        meta = json.loads((tmp_path / "qc.csv.meta.json").read_text())
        meta[-1]["run_identity"]["inputs"]["segmentation_manifest.csv"] = qc_seg_hash
        (tmp_path / "qc.csv.meta.json").write_text(json.dumps(meta))
    params = tmp_path / "params.yaml"
    params.write_text("{}")
    return argparse.Namespace(seg_root=str(seg_root), qc=str(qc_path), params=str(params),
                              out=str(tmp_path / "features.csv"), timepoints="T0,T2", workers=1, limit=0)


def test_qc_table_must_come_from_the_same_segmentation_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "get_extractor", lambda params: None)
    with pytest.raises(ValueError, match="different segmentation manifest"):
        extract.run_extraction(completed_inputs(tmp_path, qc_seg_hash="0" * 64))


def test_qc_table_must_cover_every_segmented_examination(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "get_extractor", lambda params: None)
    with pytest.raises(ValueError, match="does not cover exactly"):
        extract.run_extraction(completed_inputs(tmp_path, qc_timepoints=("T0",)))
