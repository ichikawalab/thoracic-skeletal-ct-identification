"""Input defects are detected during the all-slice inventory; measurements survive holds and resume."""
import argparse
import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.encaps import encapsulate

from common.dicom_series import (
    audit_series,
    dicom_files,
    is_uniform,
    listing,
    open_inventory,
    save_series,
    slice_offsets,
    slice_to_hu,
    sort_and_dedupe,
    uid_conflicts,
)
from common.provenance import require_completion
from tests.conftest import load_stage_script, make_ct_series

inventory = load_stage_script("01_preprocessing/inventory_dicom_series.py")
AXIAL = np.array([1, 0, 0, 0, 1, 0], dtype=float)


def audit(headers):
    for h in headers:
        h.save_as(h.filename, write_like_original=False)
    return audit_series(sorted(h.filename for h in headers), "1")[2]


# ---- Slice geometry ----

def test_offsets_are_projections_on_the_slice_normal():
    positions = np.array([[0, 0, 10.0], [0, 0, 12.5], [5, 5, 15.0]])
    assert np.allclose(slice_offsets(positions, AXIAL), [10.0, 12.5, 15.0])
    flipped = np.array([1, 0, 0, 0, -1, 0], dtype=float)  # normal points to -z
    assert np.allclose(slice_offsets(positions[:2], flipped), [-10.0, -12.5])


def test_sort_and_dedupe_orders_and_drops_duplicates():
    stats = sort_and_dedupe(np.array([12.5, 10.0, 15.0, 12.5, 17.5]))
    assert (stats.n_input, stats.n_unique, stats.n_duplicates) == (5, 4, 1)
    assert (stats.spacing_median_mm, stats.spacing_max_dev_mm, stats.extent_mm) == (2.5, 0.0, 7.5)
    assert stats.order.tolist() == [1, 0, 2, 4]


def test_uniformity_tolerance_and_single_slice():
    assert is_uniform(sort_and_dedupe(np.arange(0, 100, 2.5)))
    assert not is_uniform(sort_and_dedupe(np.array([0, 2.5, 5.0, 10.0, 12.5])))
    assert is_uniform(sort_and_dedupe(np.array([0, 2.5, 5.02, 7.5, 10.0])))  # 0.02 mm jitter is within 0.05 mm
    assert not is_uniform(sort_and_dedupe(np.array([3.0])))


def test_nearby_positions_do_not_form_a_long_duplicate_chain():
    assert sort_and_dedupe(np.array([0, 0.009, 0.018, 2.5])).order.tolist() == [0, 2, 3]


@pytest.mark.parametrize("origin", [-250.0, 100.0])
@pytest.mark.parametrize("delta,valid", [(0.05 - 1e-8, True), (0.05, True), (0.05 + 1e-8, False)])
@pytest.mark.parametrize("kind", ["spacing", "inplane", "grid"])
def test_geometry_thresholds_do_not_depend_on_coordinate_origin(tmp_path, origin, delta, valid, kind):
    headers = make_ct_series(tmp_path / "ct", n=21)
    for k, h in enumerate(headers):
        h.ImagePositionPatient = [origin, 0, origin + k * 2.5]
        if kind == "grid":  # accumulate an offset while each step remains within the spacing tolerance
            h.ImagePositionPatient[2] += max(k - 15, 0) * delta / 5
    if kind != "grid":
        headers[-1].ImagePositionPatient[0 if kind == "inplane" else 2] += delta
    result = audit(headers)
    assert (result["validation_status"] == "valid") == valid
    if not valid:
        assert result["validation_reason"] == {"spacing": "nonuniform_spacing", "inplane": "inplane_position_varies",
                                               "grid": "slice_positions_not_on_grid"}[kind]


@pytest.mark.parametrize("origin", [-250.0, 100.0])
@pytest.mark.parametrize("delta,valid", [(0.01 - 1e-8, True), (0.01, True), (0.01 + 1e-8, False)])
def test_duplicate_threshold_does_not_depend_on_coordinate_origin(tmp_path, origin, delta, valid):
    headers = make_ct_series(tmp_path / "ct")
    for k, h in enumerate(headers):
        h.ImagePositionPatient[2] = origin + k * 2.5
    duplicate = copy.deepcopy(headers[0])
    duplicate.filename = str(tmp_path / "ct" / "duplicate.dcm")
    duplicate.ImagePositionPatient[2] += delta
    result = audit([*headers, duplicate])
    assert (result["validation_status"] == "valid") == valid
    assert result["n_duplicates"] == int(valid)


# ---- Header and pixel validation ----

@pytest.mark.parametrize("tag,value,reason", [
    ("RescaleIntercept", None, "RescaleIntercept_invalid"),
    ("RescaleSlope", 0, "RescaleSlope_invalid"),
    ("RescaleSlope", float("inf"), "RescaleSlope_invalid"),
    ("PixelSpacing", [0, 1], "PixelSpacing_invalid"),
    ("Rows", 7, "Rows_varies"),
    ("ConvolutionKernel", "B50f", "ConvolutionKernel_varies"),
    ("GantryDetectorTilt", 1, "gantry_tilt_nonzero"),
    ("ImageType", ["DERIVED", "PRIMARY", "AXIAL"], "not_original_primary_axial"),
    ("RescaleType", "OD", "unsupported_HU_mapping"),
    ("MultienergyCTAcquisition", "YES", "unsupported_HU_mapping"),
    ("NumberOfFrames", 2, "multiframe_image"),
])
def test_interior_header_defect_is_rejected(tmp_path, tag, value, reason):
    headers = make_ct_series(tmp_path / "ct")
    setattr(headers[4], tag, value)
    result = audit(headers)
    assert result["validation_status"] == "rejected"
    assert result["validation_reason"] == reason


@pytest.mark.parametrize("tag,value", [("PatientID", "2"), ("PatientID", ""),
                                      ("ClinicalTrialTimePointID", "T2"), ("ClinicalTrialTimePointID", "")])
def test_interior_identity_conflict_holds_series(tmp_path, tag, value):
    headers = make_ct_series(tmp_path / "ct")
    setattr(headers[4], tag, value)
    assert audit(headers)["validation_status"] == "hold"


def test_folder_patient_mismatch_holds_entire_series(tmp_path):
    headers = make_ct_series(tmp_path / "ct")
    assert audit_series([h.filename for h in headers], "2")[2]["validation_status"] == "hold"


@pytest.mark.parametrize("tag", ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"])
def test_missing_identity_uid_holds_instead_of_rejecting_only_series(tmp_path, tag):
    headers = make_ct_series(tmp_path / "ct")
    del headers[4][tag]
    assert audit(headers)["validation_status"] == "hold"


@pytest.mark.parametrize("tag", ["BitsStored", "SamplesPerPixel", "HighBit"])
def test_missing_encoding_is_reported_not_a_program_failure(tmp_path, tag):
    headers = make_ct_series(tmp_path / "ct")
    for h in headers:
        del h[tag]
    assert audit(headers)["validation_reason"] == tag + "_invalid"


def test_stored_bits_and_signedness_may_vary_within_12_to_16_bits(tmp_path):
    headers = make_ct_series(tmp_path / "ct")
    headers[4].BitsStored, headers[4].HighBit, headers[4].PixelRepresentation = 12, 11, 1
    result = audit(headers)
    assert result["validation_status"] == "valid" and "BitsStored" in result["varied_tags"]
    headers[4].BitsStored, headers[4].HighBit = 11, 10
    assert audit(headers)["validation_reason"] == "invalid_pixel_encoding"


# PS3.5 8.1.1: receiving applications may assume neither zero nor sign extension in the unused allocated bits.
@pytest.mark.parametrize("bits,representation,word,stored", [
    (12, 0, 0x03E8, 1000), (12, 0, 0x13E8, 1000),
    (12, 1, 0xFC18, -1000), (12, 1, 0x0C18, -1000),
    (14, 0, 0xC3E8, 1000), (14, 1, 0x3C18, -1000),
    (16, 0, 0x03E8, 1000), (16, 1, 0xFC18, -1000),
])
def test_pixel_samples_come_from_the_low_bits_stored_bits(tmp_path, bits, representation, word, stored):
    ds = make_ct_series(tmp_path / "ct", n=1)[0]
    ds.BitsStored, ds.HighBit, ds.PixelRepresentation = bits, bits - 1, representation
    ds.PixelData = np.full((6, 8), word, dtype=np.uint16).tobytes()
    ds.save_as(ds.filename, write_like_original=False)
    assert (slice_to_hu(pydicom.dcmread(ds.filename)) == stored - 1024).all()  # RescaleIntercept is -1024


def test_missing_optional_tilt_does_not_reject_valid_geometry(tmp_path):
    headers = make_ct_series(tmp_path / "ct")
    for h in headers:
        del h.GantryDetectorTilt
    assert audit(headers)["validation_status"] == "valid"


def test_every_header_is_read_even_when_first_file_is_unreadable(tmp_path, monkeypatch):
    headers = make_ct_series(tmp_path / "ct")
    (tmp_path / "ct" / "000-blank.dcm").touch()
    calls = []
    read = pydicom.dcmread

    def tracked(path, **kwargs):
        calls.append(str(path))
        return read(path, **kwargs)

    monkeypatch.setattr(pydicom, "dcmread", tracked)
    row, _ = inventory.scan_series("1", "study", str(tmp_path / "ct"))
    assert len(calls) == len(headers) + 1  # every header is read; decoding waits for a fully readable series
    assert row["n_unread"] == 1 and row["validation_status"] == "incomplete"


@pytest.mark.filterwarnings("ignore::UserWarning")  # pydicom warns about UIDs cut mid-value
def test_truncated_file_at_any_length_is_recorded_not_fatal(tmp_path):
    headers = make_ct_series(tmp_path / "ct", n=2)
    path = Path(headers[1].filename)
    data = path.read_bytes()
    files = [h.filename for h in headers]
    for length in range(len(data)):
        path.write_bytes(data[:length])
        assert audit_series(files, "1")[2]["validation_status"] in ("incomplete", "hold", "rejected")


@pytest.mark.parametrize("tag", ["Rows", "Columns", "SamplesPerPixel", "BitsAllocated"])
def test_empty_required_tag_is_rejected_before_decoding(tmp_path, tag):
    headers = make_ct_series(tmp_path / "ct")
    setattr(headers[4], tag, None)
    result = audit(headers)
    assert result["validation_status"] == "rejected" and result["validation_reason"].startswith(tag)


def test_malformed_numeric_header_is_rejected(tmp_path):
    headers = make_ct_series(tmp_path / "ct")
    path = Path(headers[4].filename)
    data = path.read_bytes()
    offset = data.index(b"\x28\x00\x52\x10DS")
    size = int.from_bytes(data[offset + 6:offset + 8], "little")
    path.write_bytes(data[:offset + 8] + b"x" * size + data[offset + 8 + size:])
    row, _ = inventory.scan_series("1", "study", str(tmp_path / "ct"))
    assert row["validation_status"] == "rejected"
    assert row["validation_reason"] == "RescaleIntercept_invalid"


def test_unassignable_series_is_held_for_review(tmp_path):
    path = tmp_path / "blank.dcm"
    path.touch()
    result = audit_series([path], "1")[2]
    assert result["validation_status"] == "incomplete" and result["n_unread"] == 1


def test_compressed_slice_without_decoder_is_rejected_not_fatal(tmp_path):
    headers = make_ct_series(tmp_path / "ct")
    headers[4].file_meta.TransferSyntaxUID = pydicom.uid.JPEGLosslessSV1
    headers[4].PixelData = encapsulate([b"\xff\xd8\xff\xc3" + bytes(64)])
    headers[4]["PixelData"].is_undefined_length = True
    result = audit(headers)
    assert result["validation_status"] == "rejected"
    assert result["validation_reason"] == "unsupported_transfer_syntax"


def test_missing_interior_pixels_blocks_exam_without_hiding_later_identity(tmp_path):
    headers = make_ct_series(tmp_path / "ct")
    del headers[3].PixelData
    assert audit(headers)["validation_status"] == "incomplete"
    headers[-1].PatientID = "2"
    result = audit(headers)
    assert result["validation_status"] == "hold"
    assert result["observed_patient_ids"] == "1;2"
    assert result["extent_mm"] == 20


def test_extent_does_not_depend_on_filename_endpoints(tmp_path):
    headers = make_ct_series(tmp_path / "ct")
    for h, k in zip(headers, [4, 0, 1, 2, 3, 8, 7, 6, 5], strict=True):
        h.ImagePositionPatient[2] = -250 + k * 2.5
        h.InstanceNumber = k + 1
    result = audit(headers)
    assert result["validation_status"] == "valid"
    assert result["extent_mm"] == 20 and result["spacing_median_mm"] == 2.5


@pytest.mark.parametrize("changed,acquisition", [(False, 1), (True, 1), (True, 2)])
def test_duplicate_content_not_acquisition_number_controls_acceptance(tmp_path, changed, acquisition):
    headers = make_ct_series(tmp_path / "ct")
    duplicate = copy.deepcopy(headers[4])
    duplicate.filename = str(tmp_path / "ct" / "duplicate.dcm")
    duplicate.AcquisitionNumber = acquisition
    if changed:
        duplicate.PixelData = (duplicate.pixel_array + 100).astype(np.uint16).tobytes()
    result = audit(headers + [duplicate])
    assert result["validation_status"] == ("rejected" if changed else "valid")
    assert result["n_duplicates"] == 1
    if changed:
        assert result["validation_reason"] == "conflicting_duplicate_pixels"


def test_same_raw_pixels_with_different_rescale_are_not_duplicates(tmp_path):
    headers = make_ct_series(tmp_path / "ct")
    duplicate = copy.deepcopy(headers[4])
    duplicate.filename = str(tmp_path / "ct" / "duplicate.dcm")
    duplicate.RescaleIntercept = -1000
    assert audit(headers + [duplicate])["validation_reason"] == "conflicting_duplicate_pixels"


def test_reused_sop_uid_at_different_position_is_rejected(tmp_path):
    headers = make_ct_series(tmp_path / "ct")
    headers[4].SOPInstanceUID = headers[3].SOPInstanceUID
    headers[4].file_meta.MediaStorageSOPInstanceUID = headers[3].SOPInstanceUID
    assert audit(headers)["validation_reason"] == "SOPInstanceUID_reused_at_different_position"


def test_variable_rescale_and_exposure_are_recorded_and_allowed(tmp_path):
    headers = make_ct_series(tmp_path / "ct", varying_rescale=True)
    for k, h in enumerate(headers):
        h.XRayTubeCurrent = 50 + k
        h.Exposure = 100 + k
    result = audit(headers)
    assert result["validation_status"] == "valid" and result["flag_rescale_varies"]
    assert "XRayTubeCurrent" in result["varied_tags"]


def test_interior_drift_with_matching_endpoints_is_rejected(tmp_path):
    headers = make_ct_series(tmp_path / "ct")
    headers[4].ImagePositionPatient[0] += 1
    assert audit(headers)["validation_reason"] == "inplane_position_varies"


def test_accumulated_position_error_is_rejected(tmp_path):
    headers = make_ct_series(tmp_path / "ct", n=11)
    for k, h in enumerate(headers):
        h.ImagePositionPatient[2] += min(k, 4) * 0.04
    assert audit(headers)["validation_reason"] == "slice_positions_not_on_grid"


def test_small_position_roundoff_is_allowed(tmp_path):
    headers = make_ct_series(tmp_path / "ct")
    headers[4].ImagePositionPatient[2] += 0.02
    assert audit(headers)["validation_status"] == "valid"


# ---- Inventory checkpoint and global UID audit ----

def test_partial_inventory_cannot_be_selected_and_empty_subjects_are_counted(tmp_path):
    root = tmp_path / "dicom"
    make_ct_series(root / "1" / "study" / "series")
    (root / "2").mkdir()
    options = argparse.Namespace(root=str(root), out=str(tmp_path / "inventory.csv"), workers=1, limit=1)
    inventory.run_inventory(options)
    with pytest.raises(ValueError, match="Incomplete"):
        require_completion(options.out)
    options.limit = 0
    inventory.run_inventory(options)
    record = require_completion(options.out)
    assert record["subjects"] == ["1", "2"] and record["empty_subjects"] == ["2"]
    assert len(pd.read_csv(options.out)) == 1
    (root / "1" / "study" / "new_series").mkdir()
    with pytest.raises(ValueError, match="listing changed"):
        inventory.run_inventory(options)


def test_one_slice_measurement_can_be_checkpointed_and_resumed(tmp_path):
    root = tmp_path / "dicom"
    make_ct_series(root / "1" / "study" / "one", n=1)
    options = argparse.Namespace(root=str(root), out=str(tmp_path / "inventory.csv"), workers=1, limit=0)
    inventory.run_inventory(options)
    inventory.run_inventory(options)
    frame = pd.read_csv(options.out)
    assert frame.iloc[0].validation_status == "rejected" and frame.iloc[0].extent_mm == 0


def test_global_uid_index_includes_nonrepresentative_instances(tmp_path):
    root = tmp_path / "dicom"
    one = make_ct_series(root / "1" / "study" / "one")
    two = make_ct_series(root / "2" / "study" / "two")
    for header in two:
        header.PatientID = "2"
        header.save_as(header.filename, write_like_original=False)
    two[4].StudyInstanceUID = one[4].StudyInstanceUID
    two[4].save_as(two[4].filename, write_like_original=False)
    jobs = inventory.series_jobs(str(root), "1") + inventory.series_jobs(str(root), "2")
    db = open_inventory(tmp_path / "index.sqlite", root, jobs, ["1", "2"])
    try:
        for job in jobs:
            save_series(db, root, *inventory.scan_series(*job))
        assert uid_conflicts(db, root) == {"1/study/one", "2/study/two"}
    finally:
        db.close()
    options = argparse.Namespace(root=str(root), out=str(tmp_path / "inventory.csv"), workers=2, limit=0)
    inventory.run_inventory(options)
    first = pd.read_csv(options.out, dtype=str, keep_default_na=False).set_index("pid")
    assert first.loc["1", "validation_reason"] == "global_UID_conflict"
    assert first.loc["2", "validation_reason"] == "StudyInstanceUID_missing_or_mixed;global_UID_conflict"
    assert first.validation_status.eq("hold").all()
    inventory.run_inventory(options)
    pd.testing.assert_frame_equal(first, pd.read_csv(options.out, dtype=str, keep_default_na=False).set_index("pid"))


def test_same_owner_sop_copy_requires_matching_hu(tmp_path):
    root = tmp_path / "dicom"
    headers = make_ct_series(root / "1" / "study" / "a")
    shutil.copytree(root / "1" / "study" / "a", root / "1" / "study" / "b")
    jobs = inventory.series_jobs(str(root), "1")
    db = open_inventory(tmp_path / "index.sqlite", root, jobs, ["1"])
    try:
        for job in jobs:
            save_series(db, root, *inventory.scan_series(*job))
        assert uid_conflicts(db, root) == set()
        h = headers[4]
        for delta, expected in ((0.01 - 1e-8, set()), (0.01, set()), (0.01 + 1e-8, {"1/study/a", "1/study/b"})):
            shifted = copy.deepcopy(h)
            shifted.ImagePositionPatient[0] += delta
            shifted.save_as(root / "1" / "study" / "b" / Path(h.filename).name, write_like_original=False)
            assert uid_conflicts(db, root) == expected
        h.PixelData = (h.pixel_array + 10).astype(np.uint16).tobytes()
        h.save_as(root / "1" / "study" / "b" / Path(h.filename).name, write_like_original=False)
        assert uid_conflicts(db, root) == {"1/study/a", "1/study/b"}
    finally:
        db.close()


def test_indexed_retry_replaces_only_its_own_instances(tmp_path):
    root = tmp_path / "dicom"
    for name in ("a", "b"):
        make_ct_series(root / "1" / "study" / name)
    jobs = inventory.series_jobs(str(root), "1")
    db = open_inventory(tmp_path / "index.sqlite", root, jobs, ["1"])
    try:
        for job in jobs:
            save_series(db, root, *inventory.scan_series(*job))
        before = db.execute("SELECT * FROM instances ORDER BY path,file").fetchall()
        save_series(db, root, *inventory.scan_series(*jobs[0]))
        assert db.execute("SELECT * FROM instances ORDER BY path,file").fetchall() == before
    finally:
        db.close()


def test_listing_retains_names_sizes_and_precise_timestamps(tmp_path):
    for name, content in (("B.DCM", b"abc"), ("a.dcm", b""), ("C.dcm", b"c"), ("d.DCM", b"d"),
                          ("ignored.txt", b"other")):
        (tmp_path / name).write_bytes(content)
    (tmp_path / "directory.dcm").mkdir()
    expected = [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in dicom_files(tmp_path)]
    assert listing(tmp_path) == expected
