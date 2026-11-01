"""GPU scheduling, mask validation, per-task retry and model integrity (mocked inference)."""
import argparse
import json
import sys
import types
from concurrent.futures import Future

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from common.provenance import file_hash, mark_completion, require_completion, start_run
from tests.conftest import load_stage_script

seg = load_stage_script("01_preprocessing/run_segmentation.py")


def test_pid_range_is_inclusive_and_optional():
    assert seg.parse_pid_range("1000-2000") == (1000, 2000)
    for bad in ("2000-1000", "-1-2", "a-b", "1000", "1-2-3"):
        with pytest.raises(argparse.ArgumentTypeError):
            seg.parse_pid_range(bad)
    pids = [str(p) for p in range(1000, 1048)]
    low = [p for p in pids if seg.in_pid_range(p, (1000, 1023))]
    high = [p for p in pids if seg.in_pid_range(p, (1024, 9999))]
    assert low + high == pids and low[-1] == "1023" and high[0] == "1024"
    assert all(seg.in_pid_range(p, None) for p in pids)


def test_each_device_has_its_own_bounded_pool(tmp_path, monkeypatch):
    pools = []

    class Pool:
        def __init__(self, workers, mp_context):
            self.devices, self.closed = [], False
            assert workers == (1 if not pools else 2) and mp_context.get_start_method() == "spawn"
            pools.append(self)

        def submit(self, function, *args):
            self.devices.append(args[3])
            future = Future()
            future.set_result(function(*args))
            return future

        def shutdown(self, **kwargs):
            self.closed = True

    def segment(root, out, exam, device, code):
        return {**dict.fromkeys(seg.FIELDS, ""), "pid": exam["pid"], "timepoint": "T0", "device": device,
                "code_sha256": code, "status": "complete"}

    for i in range(9):
        (tmp_path / f"{i}.nii.gz").write_bytes(b"mock converter output")
    pd.DataFrame([{**dict.fromkeys(seg.NIFTI_FIELDS, ""), "pid": str(i), "timepoint": "T0",
                   "nifti_path": f"{i}.nii.gz", "status": "complete",
                   "nifti_sha256": file_hash(tmp_path / f"{i}.nii.gz")} for i in range(9)]).to_csv(
        tmp_path / "nifti_manifest.csv", index=False)
    start_run("convert.py", str(tmp_path / "nifti_manifest.csv"), argparse.Namespace())
    mark_completion(tmp_path / "nifti_manifest.csv", True)
    monkeypatch.setattr(seg, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(seg, "segment_exam", segment)
    monkeypatch.setattr(seg, "write_label_maps", lambda path: None)
    monkeypatch.setattr(seg, "validate_devices", lambda devices: devices)
    monkeypatch.setattr(seg, "check_patch", lambda: "test")
    monkeypatch.setattr(seg, "prepare_models", lambda root: None)
    monkeypatch.setattr(seg.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(sys, "argv", ["seg", "--nifti-root", str(tmp_path), "--out-root", str(tmp_path),
                                      "--devices", "gpu:0,gpu:1", "--workers-per-device", "2"])
    seg.main()
    assert len(pools) == 3 and all(pool.closed for pool in pools)
    assert pools[0].devices == ["gpu:0"]
    assert pools[1].devices == ["gpu:0"] * 4 and pools[2].devices == ["gpu:1"] * 4
    assert len(pd.read_csv(tmp_path / "segmentation_manifest.csv")) == 9
    assert require_completion(tmp_path / "segmentation_manifest.csv")["n_successful"] == 9


def test_mask_geometry_and_labels_are_verified(tmp_path, monkeypatch):
    maps = types.ModuleType("totalsegmentator.map_to_binary")
    maps.class_map = {"total": {116: "sternum"}, "vertebrae_pp_refined": {8: "vertebrae_T1"}}
    monkeypatch.setitem(sys.modules, maps.__name__, maps)
    image, mask = tmp_path / "ct.nii.gz", tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.uint8), np.eye(4)), image)
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[3:7, 3:7, 3:7] = 116
    nib.save(nib.Nifti1Image(data, np.eye(4)), mask)
    assert seg.validate_mask(image, mask, "sternum").max() == 116
    affine = np.eye(4)
    affine[0, 3] = 1
    nib.save(nib.Nifti1Image(data, affine), mask)
    with pytest.raises(ValueError, match="geometry"):
        seg.validate_mask(image, mask, "sternum")
    data[4, 4, 4] = 8
    nib.save(nib.Nifti1Image(data, np.eye(4)), mask)
    with pytest.raises(ValueError, match="label"):
        seg.validate_mask(image, mask, "sternum")


def test_sternum_survives_vertebral_failure_and_retry(tmp_path, monkeypatch):
    image = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.uint8), np.eye(4)), image)
    exam = dict(pid="1", timepoint="T0", nifti_path=image.name, nifti_sha256=file_hash(image))
    calls = []

    def task(image, output, device, spec):
        calls.append(spec["task"])
        if spec["task"] == "vertebrae_pp_refined" and len(calls) == 1:
            output.write_bytes(b"partial mask")
            raise RuntimeError("temporary model failure")
        nib.save(nib.Nifti1Image(np.ones((10, 10, 10), dtype=np.uint8), np.eye(4)), output)
        return 0.1

    monkeypatch.setattr(seg, "run_task", task)
    monkeypatch.setattr(seg, "validate_mask", lambda image, mask, name: nib.load(mask).get_fdata())
    out = tmp_path / "seg"
    failed = seg.segment_exam(str(tmp_path), str(out), exam, "cpu", "code-a")
    assert failed["status"] == "error" and failed["sternum_status"] == "complete"
    assert not failed["vertebrae_path"] and not list(out.rglob("*vertebrae.nii.gz"))
    success = seg.segment_exam(str(tmp_path), str(out), {**exam, "previous": failed}, "cpu", "code-a")
    assert success["status"] == "complete" and success["sternum_sha256"] == failed["sternum_sha256"]
    assert calls == ["vertebrae_pp_refined", "total", "vertebrae_pp_refined"]
    changed = seg.segment_exam(str(tmp_path), str(out), {**exam, "previous": failed}, "cpu", "code-b")
    assert changed["code_sha256"] == "code-b" and calls[-2:] == ["vertebrae_pp_refined", "total"]


def test_usage_telemetry_is_disabled_without_changing_user_config(tmp_path, monkeypatch):
    api = types.ModuleType("totalsegmentator.python_api")
    config = types.ModuleType("totalsegmentator.config")
    config.get_weights_dir = lambda: tmp_path
    checked = []

    def infer(**kwargs):
        path = seg.Path(seg.os.environ["TOTALSEG_HOME_DIR"])
        assert json.loads((path / "config.json").read_text())["send_usage_stats"] is False
        checked.append(path)

    api.totalsegmentator = infer
    monkeypatch.setitem(sys.modules, api.__name__, api)
    monkeypatch.setitem(sys.modules, config.__name__, config)
    monkeypatch.setenv("TOTALSEG_HOME_DIR", "original_user_configuration")
    seg.run_task(tmp_path / "input", tmp_path / "output", "cpu", {"task": "total"})
    assert checked and not checked[0].exists()
    assert seg.os.environ["TOTALSEG_HOME_DIR"] == "original_user_configuration"


def test_model_hash_change_requires_resolution(tmp_path, monkeypatch):
    root, out = tmp_path / "weights", tmp_path / "output"
    config = types.ModuleType("totalsegmentator.config")
    config.get_weights_dir = lambda: root
    mapping = types.ModuleType("totalsegmentator.map_tasks_config")
    mapping.TASK_ID_WEIGHTS_CONFIGS = {k: {"foldername": f"Dataset{k}_test"} for k in seg.MODEL_IDS}
    library = types.ModuleType("totalsegmentator.libs")

    def download(key):
        directory = root / mapping.TASK_ID_WEIGHTS_CONFIGS[key]["foldername"]
        directory.mkdir()
        (directory / "checkpoint_final.pth").write_bytes(b"model weights")

    library.download_pretrained_weights = download
    for module in (config, mapping, library):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    seg.prepare_models(out)
    seg.prepare_models(out)
    next(root.rglob("*.pth")).write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="weights changed"):
        seg.prepare_models(out)
