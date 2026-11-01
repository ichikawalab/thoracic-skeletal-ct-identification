"""Stage 1d: segment the vertebral bodies (task vertebrae_pp_refined) and the sternum (task total) of each volume.

Output (--out-root): <pid>/<pid>_<timepoint>_{vertebrae,sternum}.nii.gz, segmentation_manifest.csv (one row per
examination: paths, hashes, device, seconds and per-task status), label_maps.json and model_weights.json (hashes
of the pinned model files, verified on every run). Runs in the env-seg environment on one or more GPUs; a
verified task output produced by the same code on the same device is reused on resume. Requires the
TotalSegmentator patch (common/totalsegmentator_patch.py).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

from common.provenance import (
    NIFTI_FIELDS,
    SEGMENTATION_FIELDS as FIELDS,
    TIME_POINTS,
    append_records,
    completed_exams,
    file_hash,
    finish_manifest,
    output_lock,
    read_manifest,
    require_completion,
    source_hash,
    start_run,
    validate_previous,
    verified_file,
    worker_pools,
    write_csv_atomic,
    write_json,
)
from common.totalsegmentator_patch import check_patch

TASKS = {"vertebrae": {"task": "vertebrae_pp_refined"}, "sternum": {"task": "total", "roi_subset": ["sternum"]}}
# total (1.5 mm) and its 6 mm cropping model, vertebral-body refinement and vertebrae_pp.
MODEL_IDS = (291, 292, 293, 294, 295, 298, 305, 803)


def parse_pid_range(text: str) -> tuple[int, int]:
    """'MIN-MAX' -> (min, max), an inclusive range of numeric patient IDs."""
    try:
        low, high = (int(x) for x in text.split("-"))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--pid-range must look like MIN-MAX, got {text!r}") from e
    if low < 0 or high < low:
        raise argparse.ArgumentTypeError(f"--pid-range needs 0 <= MIN <= MAX, got {text!r}")
    return low, high


def in_pid_range(pid: str, pid_range: tuple[int, int] | None) -> bool:
    return pid_range is None or pid_range[0] <= int(pid) <= pid_range[1]


def run_task(image: Path, output: Path, device: str, spec: dict) -> float:
    """Run one TotalSegmentator task in this (worker) process; return wall-clock seconds."""
    from totalsegmentator.config import get_weights_dir
    from totalsegmentator.python_api import totalsegmentator

    t0 = time.perf_counter()
    previous = {key: os.environ.get(key) for key in ("TOTALSEG_HOME_DIR", "TOTALSEG_WEIGHTS_PATH")}
    weights = str(get_weights_dir().resolve())
    try:  # A temporary home directory disables telemetry without touching the user's configuration.
        with tempfile.TemporaryDirectory(prefix="totalseg-config-") as directory:
            os.environ.update(TOTALSEG_HOME_DIR=directory, TOTALSEG_WEIGHTS_PATH=weights)
            write_json(Path(directory) / "config.json",
                       {"totalseg_id": "local_pipeline", "send_usage_stats": False, "prediction_counter": 0,
                        "statistics_disclaimer_shown": True})
            totalsegmentator(input=image, output=output, ml=True, device=device, quiet=True, verbose=False,
                             nr_thr_resamp=1, nr_thr_saving=1, **spec)  # examinations already run in parallel
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return round(time.perf_counter() - t0, 1)


def prepare_models(out_root) -> None:
    """Download the model files once and verify their hashes on every run."""
    from totalsegmentator.config import get_weights_dir
    from totalsegmentator.libs import download_pretrained_weights
    from totalsegmentator.map_tasks_config import TASK_ID_WEIGHTS_CONFIGS

    path = Path(out_root) / "model_weights.json"
    root = get_weights_dir()
    directories = [root / TASK_ID_WEIGHTS_CONFIGS[key]["foldername"] for key in MODEL_IDS]
    root.mkdir(parents=True, exist_ok=True)
    if any(p.is_dir() and p not in directories and p.name != "__MACOSX" for p in root.iterdir()):  # macOS zip metadata
        raise ValueError("Use a dedicated model cache for these tasks; unrelated model directories are present")
    if not path.exists():
        for model_id in MODEL_IDS:
            download_pretrained_weights(model_id)
    snapshot = {}
    for directory in directories:
        files = sorted(p for p in directory.rglob("*") if p.suffix in (".pth", ".json", ".pkl"))
        if not any(p.suffix == ".pth" for p in files):
            raise ValueError(f"Model {directory.name} has no checkpoint")
        snapshot.update({p.relative_to(root).as_posix(): file_hash(p) for p in files})
    if not path.exists():
        write_json(path, snapshot)
    elif snapshot != json.loads(path.read_text(encoding="utf-8")):
        raise ValueError("Model weights changed or are missing; restore them or use a new output root")


def validate_mask(image, mask, name: str) -> np.ndarray:
    from totalsegmentator.map_to_binary import class_map

    reference, result = nib.load(image), nib.load(mask)
    if result.shape != reference.shape or not np.allclose(result.affine, reference.affine, atol=1e-4, rtol=0):
        raise ValueError("Mask geometry differs from input CT")
    data = np.asanyarray(result.dataobj)
    task = TASKS[name]["task"]
    allowed = {0} | {int(k) for k, v in class_map[task].items() if name != "sternum" or v == "sternum"}
    if not np.isfinite(data).all() or not set(np.unique(data)) <= allowed:
        raise ValueError("Mask contains unexpected label values")
    return data


def segment_exam(nifti_root: str, out_root: str, exam: dict, device: str, code: str) -> dict:
    """Run both tasks; a verified task output that this code produced on this device earlier is reused."""
    row = dict.fromkeys(FIELDS, "")
    row.update(pid=exam["pid"], timepoint=exam["timepoint"], nifti_path=exam["nifti_path"],
               nifti_sha256=exam["nifti_sha256"], device=device, code_sha256=code)
    image = Path(nifti_root) / exam["nifti_path"]
    out_dir = Path(out_root) / exam["pid"]
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = exam.get("previous", {})
    for name, spec in TASKS.items():
        count = "vertebra_labels" if name == "vertebrae" else "sternum_voxels"
        try:
            if (previous.get(name + "_status") == "complete" and previous.get("code_sha256") == code
                    and previous.get("device") == device and verified_file(out_root, previous, name)):
                for field in (name + "_path", name + "_sha256", name + "_status", "seconds_" + name, count):
                    row[field] = previous[field]
                continue
            with tempfile.TemporaryDirectory(dir=out_dir, prefix="seg-") as temporary:
                mask = Path(temporary) / (name + ".nii.gz")
                row["seconds_" + name] = run_task(image, mask, device, spec)
                data = validate_mask(image, mask, name)
                target = out_dir / f"{exam['pid']}_{exam['timepoint']}_{name}.nii.gz"
                os.replace(mask, target)
            row[name + "_path"] = target.relative_to(out_root).as_posix()
            row[name + "_sha256"] = file_hash(target)
            row[name + "_status"] = "complete"
            row[count] = int((np.unique(data) > 0).sum()) if name == "vertebrae" else int((data > 0).sum())
        except (OSError, RuntimeError, ValueError, nib.filebasedimages.ImageFileError) as e:
            row[name + "_status"] = "error"
            row["error"] += f"{name}: {type(e).__name__}: {e}"[:200] + "; "
    row["status"] = "error" if row["error"] else "complete"
    return row


def validate_devices(devices: list[str]) -> list[str]:
    import torch

    normalized = ["gpu:0" if device == "gpu" else device for device in devices]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate GPU assignment")
    for device in normalized:
        if device != "cpu" and (not device.startswith("gpu:") or not device[4:].isdigit()
                                or int(device[4:]) >= torch.cuda.device_count()):
            raise ValueError(f"Unavailable device: {device}")
    return normalized


def write_label_maps(out_root: str) -> None:
    from totalsegmentator.map_to_binary import class_map

    wanted = {k: v for k, v in class_map.items() if k in ("total", "vertebrae_body") or "vertebrae" in k}
    write_json(os.path.join(out_root, "label_maps.json"), wanted)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nifti-root", required=True, help="directory containing nifti_manifest.csv")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--weights-root", help="model cache used only by these tasks; default: <out-root>/models")
    ap.add_argument("--devices", default="gpu", help="comma-separated TotalSegmentator devices, e.g. gpu:0,gpu:1")
    ap.add_argument("--workers-per-device", type=int, default=1)
    ap.add_argument("--timepoints", default="T0,T2")
    ap.add_argument("--pid-range", type=parse_pid_range,
                    help="MIN-MAX: process only subjects whose numeric patient ID is in this range (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="segment only the first N examinations (trial run)")
    args = ap.parse_args()
    if args.workers_per_device < 1 or args.limit < 0 or not set(args.timepoints.split(",")) <= set(TIME_POINTS):
        ap.error("Invalid workers, limit or timepoints")
    devices = validate_devices(args.devices.split(","))
    weights = str(Path(args.weights_root or Path(args.out_root) / "models").resolve())
    previous_weights = os.environ.get("TOTALSEG_WEIGHTS_PATH")
    try:
        os.environ["TOTALSEG_WEIGHTS_PATH"] = weights  # inherited by the spawned worker processes
        with output_lock(os.path.join(args.out_root, "segmentation_manifest.csv")), output_lock(weights):
            run_segmentation(args, devices)
    finally:
        if previous_weights is None:
            os.environ.pop("TOTALSEG_WEIGHTS_PATH", None)
        else:
            os.environ["TOTALSEG_WEIGHTS_PATH"] = previous_weights


def run_segmentation(args: argparse.Namespace, devices: list[str]) -> None:
    input_manifest = os.path.join(args.nifti_root, "nifti_manifest.csv")
    require_completion(input_manifest)
    manifest_path = os.path.join(args.out_root, "segmentation_manifest.csv")
    log = start_run(__file__, manifest_path, args, {"postprocessing_sha256": check_patch()})
    log.info("TotalSegmentator %s", importlib.metadata.version("TotalSegmentator"))
    write_label_maps(args.out_root)
    converted = read_manifest(input_manifest, NIFTI_FIELDS)
    converted = converted[converted["timepoint"].isin(args.timepoints.split(","))].to_dict("records")
    exams = [r for r in converted if in_pid_range(r["pid"], args.pid_range)]
    if not all(verified_file(args.nifti_root, r, "nifti") for r in exams):
        raise ValueError("Converted input file is missing or changed; resume conversion first")
    expected_keys = {(r["pid"], r["timepoint"]) for r in exams}
    done, previous_by_key = set(), {}
    if os.path.exists(manifest_path):
        frame = read_manifest(manifest_path, FIELDS)
        previous = frame.to_dict("records")
        validate_previous(previous, exams, ("nifti_path", "nifti_sha256"))
        previous_by_key = {(r["pid"], r["timepoint"]): r for r in previous}
        done = completed_exams([r for r in previous if all(verified_file(args.out_root, r, name) for name in TASKS)])
        write_csv_atomic(frame, manifest_path)
    todo = [{**r, "previous": previous_by_key.get((r["pid"], r["timepoint"]), {})}
            for r in exams if (r["pid"], r["timepoint"]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    log.info("pid range %s: converted examinations: %d, already segmented: %d, retried: %d, todo: %d, devices: %s x %d",
             "all" if args.pid_range is None else f"{args.pid_range[0]}-{args.pid_range[1]}", len(exams), len(done),
             len(previous_by_key) - len(done), len(todo), devices, args.workers_per_device)
    if exams:
        prepare_models(args.out_root)
    code, t0, n_err = source_hash(), time.perf_counter(), 0
    with worker_pools(log) as pools, append_records(manifest_path, FIELDS) as write:
        if todo:
            # The first examination runs alone; its process (and CUDA context) exits before the device pools start.
            warmup = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
            pools.append(warmup)
            first = warmup.submit(segment_exam, args.nifti_root, args.out_root, todo[0], devices[0], code).result()
            warmup.shutdown(wait=True)
            pools.clear()
            write([first])
            n_err += bool(first["error"])
            log.info("warm-up: %s  vertebrae %ss  sternum %ss  %s", first["timepoint"], first["seconds_vertebrae"],
                     first["seconds_sternum"], first["error"])
            pools.extend(ProcessPoolExecutor(args.workers_per_device, mp_context=multiprocessing.get_context("spawn"))
                         for _ in devices)
            futures = [pools[i % len(devices)].submit(segment_exam, args.nifti_root, args.out_root, exam,
                                                     devices[i % len(devices)], code)
                       for i, exam in enumerate(todo[1:])]
            for i, future in enumerate(as_completed(futures), 2):
                row = future.result()
                write([row])
                n_err += bool(row["error"])
                if i % 50 == 0 or i == len(todo):
                    elapsed = time.perf_counter() - t0
                    log.info("%d/%d exams  %.1f min  (%.2f exam/min)  errors=%d",
                             i, len(todo), elapsed / 60, i / elapsed * 60, n_err)
    log.info("finished: %d segmented, %d errors", len(todo) - n_err, n_err)
    finish_manifest(manifest_path, FIELDS, expected_keys, limited=bool(args.limit))
    if n_err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
