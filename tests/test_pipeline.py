"""Run bookkeeping (manifests, completion, run identity, worker cleanup) and CLI integration with resume."""
import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from common import provenance
from common.provenance import (
    NIFTI_FIELDS,
    append_records,
    file_hash,
    finish_manifest,
    mark_completion,
    read_manifest,
    require_completion,
    start_run,
    terminate_children,
    verified_file,
    write_csv_atomic,
)
from tests.conftest import REPO_ROOT, load_stage_script, make_ct_series


def test_interrupted_quoted_manifest_tail_is_retried(tmp_path):
    path = tmp_path / "manifest.csv"
    row = {**dict.fromkeys(NIFTI_FIELDS, ""), "pid": "1", "timepoint": "T0", "status": "complete"}
    pd.DataFrame([row]).to_csv(path, index=False)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write('2,T0,"unfinished quoted path')
    assert read_manifest(path, NIFTI_FIELDS).pid.tolist() == ["1"]
    pd.DataFrame([row, {**row, "pid": "2", "status": "unknown"}, {**row, "pid": "3"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Incomplete record"):
        read_manifest(path, NIFTI_FIELDS)


def test_interrupted_header_or_multibyte_tail_can_be_resumed(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_bytes(b"")  # killed before the header was flushed
    assert read_manifest(path, NIFTI_FIELDS).empty
    row = {**dict.fromkeys(NIFTI_FIELDS, ""), "pid": "1", "timepoint": "T0", "status": "complete"}
    with append_records(path, NIFTI_FIELDS) as write:
        write([row])
    with open(path, "ab") as stream:  # killed inside a multibyte character of the next record
        stream.write("2,T0,,,,,,,,,,,,,,,,,,OSError: fichier spécifié".encode()[:-1])
    assert read_manifest(path, NIFTI_FIELDS).pid.tolist() == ["1"]


def test_manifests_use_lf_line_endings_on_every_platform(tmp_path):
    path = tmp_path / "manifest.csv"
    row = {**dict.fromkeys(NIFTI_FIELDS, ""), "pid": "1", "timepoint": "T0", "status": "complete"}
    write_csv_atomic(pd.DataFrame([row]), path)
    with append_records(path, NIFTI_FIELDS) as write:
        write([{**row, "pid": "2"}])
    data = path.read_bytes()
    assert b"\r" not in data and data.count(b"\n") == 3


def test_completion_requires_all_requested_units_and_no_failures(tmp_path):
    path = tmp_path / "manifest.csv"
    row = {**dict.fromkeys(NIFTI_FIELDS, ""), "pid": "1", "timepoint": "T0", "code_sha256": "c1", "status": "complete"}
    expected = {("1", "T0"), ("2", "T0")}
    start_run("stage.py", str(path), argparse.Namespace())
    for rows, limited, complete in (([row], False, False),
                                   ([row, {**row, "pid": "2", "status": "error", "error": "failed"}], False, False),
                                   ([row, {**row, "pid": "2"}], True, False),
                                   ([row, {**row, "pid": "2"}], False, True)):
        pd.DataFrame(rows).to_csv(path, index=False)
        assert finish_manifest(path, NIFTI_FIELDS, expected, limited=limited) is complete
        if complete:
            assert require_completion(path)["n_successful"] == 2
            assert require_completion(path)["code_versions"] == ["c1"]
        else:
            with pytest.raises(ValueError, match="Incomplete"):
                require_completion(path)


def test_missing_changed_and_escaping_manifest_files(tmp_path):
    file = tmp_path / "image.nii.gz"
    file.write_bytes(b"published")
    row = {"nifti_path": file.name, "nifti_sha256": file_hash(file)}
    assert verified_file(tmp_path, row, "nifti")
    file.write_bytes(b"incomplete")
    assert not verified_file(tmp_path, row, "nifti")
    file.unlink()
    assert not verified_file(tmp_path, row, "nifti")
    with pytest.raises(ValueError, match="escapes"):
        verified_file(tmp_path, {**row, "nifti_path": "../outside.nii.gz"}, "nifti")


def test_verified_file_hashes_a_file_once_until_it_changes(tmp_path, monkeypatch):
    file = tmp_path / "image.nii.gz"
    file.write_bytes(b"published")
    row = {"nifti_path": file.name, "nifti_sha256": file_hash(file)}
    assert verified_file(tmp_path, row, "nifti")
    monkeypatch.setattr(provenance, "file_hash", lambda path: pytest.fail("an unchanged file must not be hashed again"))
    assert verified_file(tmp_path, row, "nifti")
    provenance._VERIFIED.clear()  # another process reads the index from disk
    assert verified_file(tmp_path, row, "nifti")
    os.utime(file, ns=(0, 0))  # a rewritten file is hashed again
    monkeypatch.setattr(provenance, "file_hash", lambda path: "other")
    assert not verified_file(tmp_path, row, "nifti")


def test_completed_output_must_be_unchanged_to_resume(tmp_path):
    table = str(tmp_path / "table.csv")
    args = argparse.Namespace(out=table, workers=1)
    Path(table).write_text("value\n1\n")
    start_run("stage.py", table, args)
    mark_completion(table, True)
    start_run("stage.py", table, args)  # an unchanged completed table resumes
    mark_completion(table, True)
    Path(table).write_text("value\n999999\n")
    with pytest.raises(ValueError, match="Completed output changed"):
        start_run("stage.py", table, args)


def test_run_identity_ignores_paths_and_runtime_settings_but_not_parameters(tmp_path):
    out = str(tmp_path / "stage")
    args = argparse.Namespace(root=str(tmp_path), out=out, workers=1, limit=0, timepoints="T0,T2")
    start_run("stage.py", out, args)
    start_run("stage.py", out, argparse.Namespace(**{**vars(args), "root": str(tmp_path) + os.sep, "workers": 8}))
    with pytest.raises(ValueError, match="settings changed \\(parameters\\)"):
        start_run("stage.py", out, argparse.Namespace(**{**vars(args), "timepoints": "T0"}))


def test_code_change_resumes_and_is_recorded_per_run(tmp_path, monkeypatch):
    out = str(tmp_path / "stage")
    args = argparse.Namespace(out=out, workers=1, limit=0)
    start_run("stage.py", out, args)
    monkeypatch.setattr(provenance, "source_hash", lambda: "b" * 64)
    start_run("stage.py", out, args)  # a code change appends a run instead of refusing
    runs = json.loads(Path(out + ".meta.json").read_text(encoding="utf-8"))
    assert [run["code_sha256"] == "b" * 64 for run in runs] == [False, True]


@pytest.mark.parametrize("stage,input_arg", [("01_preprocessing/run_segmentation.py", "nifti_root"),
                                           ("02_segmentation_qc/qc_segmentation.py", "seg_root"),
                                           ("03_feature_extraction/extract_shape_features.py", "seg_root")])
def test_downstream_rejects_unfinished_upstream(tmp_path, stage, input_arg):
    module = load_stage_script(stage)
    args = argparse.Namespace(**{input_arg: str(tmp_path)})
    with pytest.raises(ValueError, match="completion record"):
        if input_arg == "nifti_root":
            module.run_segmentation(args, ["cpu"])
        elif "qc_segmentation" in stage:
            module.run_qc(args)
        else:
            module.run_extraction(args)


def _sleep_forever():
    while True:
        time.sleep(1)


def test_terminate_children_kills_spawned_child():
    child = multiprocessing.get_context("spawn").Process(target=_sleep_forever)
    child.start()
    assert child.is_alive() and terminate_children() >= 1
    child.join(timeout=10)
    assert not child.is_alive()


def test_preprocessing_cli_and_resume(tmp_path):
    repo = Path(REPO_ROOT)
    dicom = tmp_path / "dicom"
    make_ct_series(dicom / "1" / "study" / "series", varying_rescale=True)
    inventory, selection, output = tmp_path / "inventory.csv", tmp_path / "selection", tmp_path / "nifti"
    kernels = repo / "01_preprocessing/kernel_classes.csv"

    def run(script, *args):
        result = subprocess.run([sys.executable, str(repo / "01_preprocessing" / script), *map(str, args)],
                                cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert result.returncode == 0, result.stdout

    for _ in range(2):
        run("inventory_dicom_series.py", "--root", dicom, "--out", inventory, "--workers", 1)
    assert len(pd.read_csv(inventory)) == 1
    run("select_series.py", "--inventory", inventory, "--kernel-table", kernels, "--out-dir", selection)
    selected = pd.read_csv(selection / "exam_selection.csv")
    assert len(selected) == 1 and selected.iloc[0]["status"] == "selected"
    assert selected.iloc[0]["extent_mm"] == 20
    summary = pd.read_csv(selection / "kernel_summary.csv")
    assert summary[["manufacturer_group", "ConvolutionKernel", "kernel_class", "n_series"]].values.tolist() == [
        ["SIEMENS", "B30f", "soft", 1]]
    for _ in range(2):
        run("convert_to_nifti.py", "--root", dicom, "--selection-dir", selection, "--out-root", output,
            "--workers", 1)
    manifest = pd.read_csv(output / "nifti_manifest.csv", keep_default_na=False)
    assert len(manifest) == 1 and manifest.iloc[0]["error"] == ""
    assert (output / manifest.iloc[0]["nifti_path"]).exists()
    record = require_completion(output / "nifti_manifest.csv")
    assert record["n_successful"] == 1 and record["code_versions"] == [manifest.iloc[0]["code_sha256"]]
    assert len(record["code_sha256"]) == 64 and record["code_sha256"] == record["code_versions"][0]
    assert (output / "verified_files.tsv").read_text().split("\t")[0] == manifest.iloc[0]["nifti_path"]
