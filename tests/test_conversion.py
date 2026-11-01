"""dcm2niix output is verified voxel by voxel against the DICOM source before it is published."""
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from common.dicom_series import audit_series, inspect_geometry
from tests.conftest import load_stage_script, make_ct_series

convert = load_stage_script("01_preprocessing/convert_to_nifti.py")


def test_version_identity_ignores_optional_pigz_advice(monkeypatch):
    for advice in ("", "Compression will be faster with pigz.exe\n"):
        monkeypatch.setattr(convert.subprocess, "run", lambda *args, advice=advice, **kwargs:
                            subprocess.CompletedProcess(args, 3, advice + "dcm2niiX version banner\nv1.0.20260724\n"))
        assert convert.converter_version() == "v1.0.20260724"


def test_failed_version_command_is_not_accepted(monkeypatch):
    monkeypatch.setattr(convert.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, 1, "v1.0.20260724\n"))
    with pytest.raises(subprocess.CalledProcessError):
        convert.converter_version()


@pytest.mark.parametrize("orientation", [(1, 0, 0, 0, 1, 0), (0, 1, 0, -1, 0, 0), (1, 0, 0, 0, -1, 0)])
@pytest.mark.parametrize("varying_rescale", [False, True])
def test_dcm2niix_preserves_every_voxel_in_physical_space(tmp_path, orientation, varying_rescale):
    headers = make_ct_series(tmp_path / "source", varying_rescale=varying_rescale, orientation=orientation)
    output = tmp_path / "out.nii.gz"
    result = convert.convert_series(str(tmp_path), "source", str(output), "1", "T0")
    assert result["n_unique_slices"] == len(headers)
    image = nib.load(output)
    pixels = image.get_fdata()
    yy, xx = np.indices((6, 8))
    for ds in headers:
        positions = (np.array(ds.ImagePositionPatient)[:, None]
                     + np.outer(orientation[:3], xx.ravel() * 0.8) + np.outer(orientation[3:], yy.ravel() * 0.7))
        ras = positions * np.array([-1, -1, 1])[:, None]
        ijk = np.linalg.inv(image.affine) @ np.vstack([ras, np.ones(48)])
        assert np.allclose(ijk[:3], np.rint(ijk[:3]), atol=1e-4)
        actual = pixels[tuple(np.rint(ijk[:3]).astype(int))].reshape(6, 8)
        expected = ds.pixel_array.astype(float) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        assert np.allclose(actual, expected, atol=0.01, rtol=0)


def test_identical_duplicate_is_removed(tmp_path):
    headers = make_ct_series(tmp_path / "source")
    shutil.copyfile(headers[4].filename, tmp_path / "source" / "duplicate.dcm")
    result = convert.convert_series(str(tmp_path), "source", str(tmp_path / "out.nii.gz"), "1", "T0")
    assert (result["n_files"], result["n_unique_slices"], result["n_duplicates"]) == (10, 9, 1)
    assert not (tmp_path / "out.nii.gz.dcm2niix.log").exists()  # converter logs are kept only for failures


def test_hu_values_above_int16_range_are_not_clipped(tmp_path):
    for ds in make_ct_series(tmp_path / "source"):
        ds.PixelData = np.full((6, 8), 50000, dtype=np.uint16).tobytes()
        ds.save_as(ds.filename, write_like_original=False)
    output = tmp_path / "out.nii.gz"
    convert.convert_series(str(tmp_path), "source", str(output), "1", "T0")
    assert np.allclose(nib.load(output).get_fdata(), 48976)


ENCODINGS = {  # BitsStored, HighBit, PixelRepresentation, RescaleIntercept, stored-value range
    "S16neg": (16, 15, 1, 0.0, (-1000, 1500)),  # signed with negative stored values
    "U12": (12, 11, 0, -2048.0, (0, 4095)),
    "S12": (12, 11, 1, 0.0, (-2048, 2047)),
    "U16": (16, 15, 0, -1024.0, (0, 50000)),  # unsigned above the int16 range
}


def encode(ds, name, k, intercept=None):
    """Rewrite one slice with known stored values; return its analytic HU array."""
    bits, high_bit, representation, default_intercept, (low, high) = ENCODINGS[name]
    intercept = default_intercept if intercept is None else intercept
    values = np.rint(np.linspace(low + k, high - k, 48)).reshape(6, 8).astype(np.int16 if representation else np.uint16)
    ds.BitsStored, ds.HighBit, ds.PixelRepresentation = bits, high_bit, representation
    ds.RescaleSlope, ds.RescaleIntercept = 1.0, intercept
    ds.PixelData = values.tobytes()
    ds.save_as(ds.filename, write_like_original=False)
    return values.astype(float) + intercept


# With one rescale for every slice dcm2niix converts integers using the first slice's encoding, which is why the
# converter receives uniformly encoded copies; the other cases exercise its per-slice float path.
@pytest.mark.parametrize("majority,minority,index,shared_intercept", [
    ("S12", "S12", 0, None), ("S16neg", "U12", 0, None), ("U16", "U12", 0, None), ("U12", "S16neg", 0, None),
    ("U16", "S16neg", 4, None), ("S16neg", "S12", 8, None),
    ("U12", "S12", 0, -1024.0), ("S12", "U16", 8, -1024.0), ("S16neg", "U16", 0, -1024.0),
    ("U16", "S16neg", 4, -1024.0), ("U12", "U16", 4, -1024.0), ("S16neg", "S12", 0, -1024.0),
])
def test_mixed_pixel_encodings_convert_to_analytic_hu(tmp_path, majority, minority, index, shared_intercept):
    headers = make_ct_series(tmp_path / "source")
    names = [majority] * len(headers)
    names[index] = minority
    expected = [encode(ds, name, k, shared_intercept) for k, (ds, name) in enumerate(zip(headers, names, strict=True))]
    output = tmp_path / "out.nii.gz"
    convert.convert_series(str(tmp_path), "source", str(output), "1", "T0")
    image = nib.load(output)
    reference = convert.reference_affine(headers[0], 2.5)
    orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine),
                                                  nib.orientations.io_orientation(reference))
    data = np.asanyarray(image.as_reoriented(orientation).dataobj)
    assert np.array_equal(data, np.stack([e.T for e in expected], axis=-1))


# dcm2niix reads 12-bit samples per BitsStored, so a naive 16-bit read would either fail verification or, when
# the conversion copies are rewritten, agree with itself on the wrong HU.
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("representation,word,stored", [(1, 0x0C18, -1000), (0, 0x13E8, 1000)])
def test_unused_pixel_bits_do_not_change_the_published_hu(tmp_path, mixed, representation, word, stored):
    for k, ds in enumerate(make_ct_series(tmp_path / "source")):
        ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 12, 11, representation
        ds.PixelData = np.full((6, 8), word, dtype=np.uint16).tobytes()
        if mixed and k == 2:  # a second encoding of the same samples forces uniformly encoded conversion copies
            ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 15, 1
            ds.PixelData = np.full((6, 8), stored, dtype=np.int16).tobytes()
        ds.save_as(ds.filename, write_like_original=False)
    output = tmp_path / "out.nii.gz"
    result = convert.convert_series(str(tmp_path), "source", str(output), "1", "T0")
    expected = stored - 1024  # RescaleIntercept is -1024
    assert (nib.load(output).get_fdata() == expected).all()
    assert (result["hu_min"], result["hu_max"]) == (expected, expected)


def test_1_7_mm_spacing_at_large_coordinates_is_accepted(tmp_path):
    for k, h in enumerate(make_ct_series(tmp_path / "source", n=175)):
        h.ImagePositionPatient[2] = 1100 + 1.7 * k
        h.SliceThickness = 2
        h.save_as(h.filename, write_like_original=False)
    result = convert.convert_series(str(tmp_path), "source", str(tmp_path / "out.nii.gz"), "1", "T0")
    assert result["max_position_error_mm"] <= 0.05


def test_invalid_rescale_is_rejected_before_conversion(tmp_path, monkeypatch):
    headers = make_ct_series(tmp_path / "source")
    del headers[4].RescaleIntercept
    headers[4].save_as(headers[4].filename, write_like_original=False)
    monkeypatch.setattr(convert.subprocess, "run", lambda *a, **kw: pytest.fail("converter must not run"))
    with pytest.raises(ValueError, match="RescaleIntercept_invalid"):
        convert.convert_series(str(tmp_path), "source", str(tmp_path / "out.nii.gz"), "1", "T0")


def test_changed_source_is_rejected_before_conversion(tmp_path):
    headers = make_ct_series(tmp_path / "ct")
    fingerprint = audit_series([h.filename for h in headers], "1")[2]["source_sha256"]
    headers[4].PixelData = (headers[4].pixel_array + 1).astype(np.uint16).tobytes()
    headers[4].save_as(headers[4].filename, write_like_original=False)
    with pytest.raises(ValueError, match="source_changed"):
        convert.convert_series(str(tmp_path), "ct", str(tmp_path / "out.nii.gz"), "1", "T0",
                               expected_source=fingerprint)


def test_converter_failure_does_not_publish_volume(tmp_path, monkeypatch):
    headers = make_ct_series(tmp_path / "source")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["dcm2niix"])

    monkeypatch.setattr(convert.subprocess, "run", fail)
    fingerprint = audit_series([h.filename for h in headers], "1")[2]["source_sha256"]
    row = convert.convert_exam(str(tmp_path), str(tmp_path / "out"),
                               dict(pid="1", timepoint="T0", series_relpath="source",
                                    SeriesInstanceUID=str(headers[0].SeriesInstanceUID), source_sha256=fingerprint))
    assert "CalledProcessError" in row["error"] and not row["nifti_path"]
    assert not list((tmp_path / "out").rglob("*.nii.gz"))


def test_multiple_converter_outputs_are_rejected(tmp_path, monkeypatch):
    make_ct_series(tmp_path / "source")

    def run(command, **kwargs):
        work = Path(command[command.index("-o") + 1])
        (work / "volume.nii.gz").touch()
        (work / "volume_Tilt.nii.gz").touch()
        return subprocess.CompletedProcess(command, 0, "converter output", "")

    monkeypatch.setattr(convert.subprocess, "run", run)
    with pytest.raises(ValueError, match="Expected one NIfTI"):
        convert.convert_series(str(tmp_path), "source", str(tmp_path / "out.nii.gz"), "1", "T0")
    assert not (tmp_path / "out.nii.gz").exists()
    assert (tmp_path / "out.nii.gz.dcm2niix.log").read_text() == "converter output"


def test_unreadable_converter_output_is_an_error_with_its_log(tmp_path, monkeypatch):
    headers = make_ct_series(tmp_path / "source")

    def run(command, **kwargs):
        (Path(command[command.index("-o") + 1]) / "volume.nii.gz").write_bytes(b"not a NIfTI file")
        return subprocess.CompletedProcess(command, 0, "converter output", "")

    monkeypatch.setattr(convert.subprocess, "run", run)
    fingerprint = audit_series([h.filename for h in headers], "1")[2]["source_sha256"]
    row = convert.convert_exam(str(tmp_path), str(tmp_path / "out"),
                               dict(pid="1", timepoint="T0", series_relpath="source",
                                    SeriesInstanceUID=str(headers[0].SeriesInstanceUID), source_sha256=fingerprint))
    assert row["status"] == "error" and row["dcm2niix_log"] == "1/1_T0.nii.gz.dcm2niix.log"
    assert (tmp_path / "out" / row["dcm2niix_log"]).read_text() == "converter output"


def test_output_validation_rejects_changed_hu_and_geometry(tmp_path):
    headers = make_ct_series(tmp_path / "source")
    stats, _ = inspect_geometry(headers)
    files = [h.filename for h in headers]
    pixels = np.stack([h.pixel_array.astype(float).T * float(h.RescaleSlope) + float(h.RescaleIntercept)
                       for h in headers], axis=-1)
    affine = convert.reference_affine(headers[0], 2.5)
    path = tmp_path / "wrong.nii.gz"
    nib.save(nib.Nifti1Image(pixels, affine), path)
    convert.validate_volume(path, files, headers, stats)
    corrupted = pixels.copy()
    corrupted[:, :, 2] += 100
    nib.save(nib.Nifti1Image(corrupted, affine), path)
    with pytest.raises(ValueError, match="NIfTI_HU_mismatch_slice_2"):
        convert.validate_volume(path, files, headers, stats)
    affine[0, 0] += 0.01
    nib.save(nib.Nifti1Image(pixels, affine), path)
    with pytest.raises(ValueError, match="NIfTI_corner_positions_mismatch"):
        convert.validate_volume(path, files, headers, stats)


@pytest.mark.parametrize("origin", [-250.0, 100.0])
@pytest.mark.parametrize("delta,valid", [(0.05 - 1e-8, True), (0.05, True), (0.05 + 1e-8, False)])
def test_volume_position_threshold_includes_the_boundary(tmp_path, origin, delta, valid):
    headers = make_ct_series(tmp_path / "source")
    for k, h in enumerate(headers):
        h.PixelSpacing = [1, 1]
        h.ImagePositionPatient[2] = origin + delta + k * 2.5
        h.save_as(h.filename, write_like_original=False)
    stats, _ = inspect_geometry(headers)
    pixels = np.stack([h.pixel_array.astype(float).T - 1024 for h in headers], axis=-1)
    affine = convert.reference_affine(headers[0], 2.5)
    affine[2, 3] = origin  # exactly representable in the NIfTI-1 float32 affine
    path = tmp_path / "shifted.nii.gz"
    nib.save(nib.Nifti1Image(pixels, affine), path)
    files = [h.filename for h in headers]
    if valid:
        convert.validate_volume(path, files, headers, stats)
    else:
        with pytest.raises(ValueError, match="NIfTI_corner_positions_mismatch"):
            convert.validate_volume(path, files, headers, stats)


def test_unexpected_program_error_is_not_an_exclusion(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("program defect")

    monkeypatch.setattr(convert, "convert_series", fail)
    with pytest.raises(RuntimeError, match="program defect"):
        convert.convert_exam(str(tmp_path), str(tmp_path / "out"),
                             dict(pid="1", timepoint="T0", series_relpath="source",
                                  SeriesInstanceUID="uid", source_sha256="hash"))


def test_rounded_positions_are_converted_on_the_median_grid(tmp_path):
    """Positions rounded to 0.001 mm around a 1.8 mm pitch: any adjacent pair misstates the pitch by 0.001 mm."""
    for k, h in enumerate(make_ct_series(tmp_path / "source", n=121)):
        h.ImagePositionPatient[2] = round(1493.4 + 1.8 * k + (0.001 if k % 2 else 0), 3)
        h.SliceThickness = 2
        h.save_as(h.filename, write_like_original=False)
    output = tmp_path / "out.nii.gz"
    result = convert.convert_series(str(tmp_path), "source", str(output), "1", "T0")
    assert result["max_position_error_mm"] <= 0.05
    assert abs(nib.load(output).affine[2, 2] - 1.8) < 1e-3


def test_regular_series_is_converted_from_its_original_files(tmp_path, monkeypatch):
    make_ct_series(tmp_path / "source")
    commands, run = [], convert.subprocess.run

    def record(command, **kwargs):
        commands.append(command)
        return run(command, **kwargs)

    monkeypatch.setattr(convert.subprocess, "run", record)
    convert.convert_series(str(tmp_path), "source", str(tmp_path / "out.nii.gz"), "1", "T0")
    assert Path(commands[-1][-1]) == (tmp_path / "source").resolve()
