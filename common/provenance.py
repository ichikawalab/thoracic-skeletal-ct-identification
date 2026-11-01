"""Run bookkeeping shared by every stage: hashes, atomic files, completion records, locks, run metadata,
append-only manifests with per-unit resume, and worker pools.

Each stage publishes one table; ``start_run`` appends a run record to <table>.meta.json (code hash, git state,
input hashes, settings, package versions) and ``mark_completion``/``finish_manifest`` close it with the table's
hash, which ``require_completion`` checks before the next stage starts. Only the standard library and pandas are
imported so that the Python 3.9 feature-extraction environment can use this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import multiprocessing
import os
import platform
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TIME_POINTS = ("T0", "T1", "T2")  # NLST screening rounds (ClinicalTrialTimePointID)
THORACIC = [f"vertebrae_T{i}" for i in range(1, 13)]
STRUCTURES = ["sternum", *THORACIC]
SHAPE_FEATURES = [  # PyRadiomics shape features, in the order of the feature and matching tables
    "MeshVolume", "VoxelVolume", "SurfaceArea", "SurfaceVolumeRatio", "Sphericity",
    "Maximum3DDiameter", "Maximum2DDiameterSlice", "Maximum2DDiameterColumn", "Maximum2DDiameterRow",
    "MajorAxisLength", "MinorAxisLength", "LeastAxisLength", "Elongation", "Flatness",
]
TRACKED_PACKAGES = ("pydicom", "numpy", "pandas", "nibabel", "pyradiomics", "SimpleITK", "dcm2niix",
                    "TotalSegmentator", "torch", "torchvision", "nnunetv2", "scipy", "matplotlib")
# Paths are verified by content hashes and runtime settings do not change outputs; neither identifies a run.
PATH_ARGS = {"root", "out", "out_dir", "out_root", "inventory", "kernel_table", "selection_dir", "nifti_root",
             "seg_root", "qc", "params", "clinical_screening", "weights_root", "staging_dir", "features",
             "matching_dir", "participants"}
RUNTIME_ARGS = {"limit", "workers", "workers_per_device", "devices", "pid_range"}  # partition the work, not its content
NIFTI_FIELDS = ["pid", "timepoint", "nifti_path", "series_relpath", "SeriesInstanceUID", "source_sha256",
                "n_files", "n_unique_slices", "n_duplicates", "rows", "cols", "voxel_x_mm", "voxel_y_mm", "voxel_z_mm",
                "hu_min", "hu_max", "max_position_error_mm", "dcm2niix_log", "seconds", "error", "nifti_sha256",
                "code_sha256", "status"]
SEGMENTATION_FIELDS = ["pid", "timepoint", "nifti_path", "vertebrae_path", "sternum_path", "device",
                       "seconds_vertebrae", "seconds_sternum", "vertebra_labels", "sternum_voxels", "error",
                       "nifti_sha256", "vertebrae_sha256", "sternum_sha256",
                       "vertebrae_status", "sternum_status", "code_sha256", "status"]


# ---- Files, hashes and completion records ----

def file_hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def atomic_path(path):
    """Yield a temporary file beside ``path`` that replaces it only after the block succeeds."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        yield Path(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temporary, temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def write_csv_atomic(table: pd.DataFrame, path) -> None:
    with atomic_path(path) as temporary, temporary.open("w", newline="", encoding="utf-8") as stream:
        table.to_csv(stream, index=False, lineterminator="\n")
        stream.flush()
        os.fsync(stream.fileno())


def mark_completion(path, complete: bool, **details) -> None:
    """Record in the latest run of ``<path>.meta.json`` whether the stage completed, bound to the table's hash.

    ``start_run`` appended that run before any output was written, so the file exists.
    """
    meta_path = str(path) + ".meta.json"
    with open(meta_path, encoding="utf-8") as stream:
        runs = json.load(stream)
    runs[-1].update({"complete": bool(complete), **details})
    if complete:
        runs[-1]["sha256"] = file_hash(path)
    write_json(meta_path, runs)


def require_completion(path) -> dict:
    """The latest run record of ``<path>.meta.json``; raise unless that run completed and the table is unchanged."""
    meta_path = str(path) + ".meta.json"
    if not os.path.isfile(meta_path):
        raise ValueError("Incomplete stage: completion record is missing")
    with open(meta_path, encoding="utf-8") as stream:
        record = json.load(stream)[-1]
    if not record.get("complete") or record.get("sha256") != file_hash(path):
        raise ValueError("Incomplete or changed input; finish the preceding stage first")
    return record


@contextmanager
def output_lock(path):
    """Exclusive lock on ``<path>.lock``; the OS releases it when the process exits, including after a crash."""
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


# ---- Run metadata and logging ----

def git_state() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "", None
    return {"git_commit": commit, "git_dirty": dirty}


def package_versions() -> dict:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def source_hash() -> str:
    """Hash over every source, configuration and lock file of the pipeline."""
    hashes = {}
    for directory in ("common", "01_preprocessing", "02_segmentation_qc", "03_feature_extraction", "04_matching",
                      "05_evaluation", "environments"):
        for parent, directories, files in os.walk(REPO_ROOT / directory):
            directories[:] = [name for name in directories if not name.startswith(".") and name != "__pycache__"]
            for name in files:
                if Path(name).suffix in (".py", ".yaml", ".csv", ".toml", ".lock"):
                    hashes[Path(parent, name).relative_to(REPO_ROOT).as_posix()] = file_hash(Path(parent, name))
    for name in ("pyproject.toml", "uv.lock"):
        hashes[name] = file_hash(REPO_ROOT / name)
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def input_hashes(args: argparse.Namespace) -> dict:
    """Hashes of the input tables named by the arguments, including the preceding stage's published table."""
    inputs = {key: file_hash(value) for key, value in vars(args).items()
              if key in ("inventory", "kernel_table", "qc", "params", "clinical_screening", "features", "participants")
              and value}
    for key, names in {"selection_dir": ["exam_selection.csv"], "nifti_root": ["nifti_manifest.csv"],
                       "seg_root": ["segmentation_manifest.csv", "label_maps.json"],
                       "matching_dir": ["matching_results.csv"]}.items():
        if hasattr(args, key):
            inputs.update({name: file_hash(Path(getattr(args, key), name)) for name in names})
    return inputs


def write_run_metadata(table: str, script: str, args: argparse.Namespace, extra: dict) -> None:
    """Append an incomplete run to ``<table>.meta.json``.

    Resuming is refused when inputs, settings or packages changed, or when the table of a completed run was
    modified afterwards. Each run records its code hash; manifests record it per row, so a table finished after
    a code change stays attributable.
    """
    identity = {"inputs": input_hashes(args),
                "parameters": {k: v for k, v in vars(args).items() if k not in PATH_ARGS | RUNTIME_ARGS},
                "python": platform.python_version(), "packages": package_versions(), "extra": extra}
    identity = json.loads(json.dumps(identity))  # Compare in JSON form, as stored (tuples become lists).
    meta_path = table + ".meta.json"
    runs = []
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as stream:
            runs = json.load(stream)
        previous = runs[-1]
        if previous.get("run_identity", {}) != identity:
            changed = ", ".join(k for k, v in identity.items() if previous.get("run_identity", {}).get(k) != v)
            raise ValueError(f"Input or settings changed ({changed}); use a new output path")
        if previous.get("complete") and (not os.path.isfile(table) or file_hash(table) != previous["sha256"]):
            raise ValueError("Completed output changed since its run; use a new output path")
    runs.append({"complete": False, "script": os.path.basename(script), "code_sha256": source_hash(),
                 **git_state(),
                 "platform": platform.platform(), "hostname": platform.node(),
                 "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017 (Python 3.9)
                 "args": vars(args), "run_identity": identity})
    write_json(meta_path, runs)


def setup_logging(log_path: str, name: str) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    for handler in log.handlers[:]:
        handler.close()
        log.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


def start_run(script: str, table: str, args: argparse.Namespace, extra: dict | None = None) -> logging.Logger:
    """Open ``<table>.log`` and append an incomplete run to ``<table>.meta.json`` before any output is written."""
    os.makedirs(os.path.dirname(os.path.abspath(table)), exist_ok=True)
    log = setup_logging(table + ".log", os.path.basename(script))
    write_run_metadata(table, script, args, extra or {})
    return log


# ---- Append-only manifests with per-unit resume ----

def read_checkpoint(path, fields: list, statuses) -> pd.DataFrame:
    """Read status-terminated records; an interrupted final record is dropped so that its unit is retried."""
    rows = []
    # A tail cut inside a multibyte character decodes to replacement characters and is dropped like any other
    # interrupted record; complete records are valid UTF-8 and unaffected.
    with open(path, newline="", encoding="utf-8", errors="replace") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if header is None:  # interrupted before the header was flushed: no record exists
            return pd.DataFrame(columns=fields)
        if header != fields:
            raise ValueError("Manifest columns differ; use a new output root")
        try:
            for values in reader:
                rows.append(values)
        except csv.Error as exc:  # an unterminated quoted field at the end of the file
            if str(exc) != "unexpected end of data":
                raise ValueError("Malformed checkpoint CSV") from exc
    if rows and (len(rows[-1]) != len(fields) or rows[-1][-1] not in statuses):
        rows.pop()
    if any(len(row) != len(fields) or row[-1] not in statuses for row in rows):
        raise ValueError("Incomplete record inside checkpoint")
    return pd.DataFrame(rows, columns=fields)


def read_manifest(path, fields, keys=("pid", "timepoint"), statuses=("complete", "error")) -> pd.DataFrame:
    """Latest record per key."""
    frame = read_checkpoint(path, fields, statuses)
    if frame[list(keys)].eq("").any().any():
        raise ValueError("Manifest has missing examination keys")
    return frame.drop_duplicates(list(keys), keep="last")


@contextmanager
def append_records(path, fields):
    """Yield a writer that flushes records as each unit finishes, so an interrupted run can resume."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        if new:
            writer.writeheader()
            stream.flush()  # an interruption before the first record then leaves a readable header

        def write(rows):
            writer.writerows(rows)
            stream.flush()

        yield write


def finish_manifest(path, fields, expected_keys, keys=("pid", "timepoint"), statuses=("complete", "error"),
                    *, limited=False) -> bool:
    """Compact the manifest to its latest records; complete only when every expected unit succeeded."""
    latest = read_manifest(path, fields, keys, statuses)
    write_csv_atomic(latest, path)
    successful = latest["status"].ne("error") & latest["error"].eq("")
    actual = set(latest[list(keys)].itertuples(index=False, name=None))
    complete = not limited and actual == set(expected_keys) and bool(successful.all())
    mark_completion(path, complete, n_expected=len(expected_keys), n_successful=int(successful.sum()),
                    code_versions=sorted(set(latest["code_sha256"])))
    return complete


def validate_previous(previous, expected, identity_fields) -> None:
    """Reject a manifest written for different examinations or inputs."""
    by_key = {(row["pid"], row["timepoint"]): row for row in expected}
    for row in previous:
        match = by_key.get((row["pid"], row["timepoint"]))
        if match is None or any(row[field] != match[field] for field in identity_fields):
            raise ValueError("Manifest identity differs from inputs; use a new output root")


def completed_exams(rows) -> set:
    return {(row["pid"], row["timepoint"]) for row in rows if row["status"] == "complete" and row["error"] == ""}


_VERIFIED: dict[str, dict] = {}  # per output root: relative path -> (size, mtime_ns, sha256) of hashed files


def verified_index(root: Path) -> tuple[Path, dict]:
    """The append-only record of verified files below ``root``, read once per process."""
    path = root / "verified_files.tsv"
    if str(path) not in _VERIFIED:
        index = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) == 4:
                    index[parts[0]] = (int(parts[1]), int(parts[2]), parts[3])
        _VERIFIED[str(path)] = index
    return path, _VERIFIED[str(path)]


def verified_file(root, row, name) -> bool:
    """True when ``<name>_path`` exists below ``root`` with the recorded ``<name>_sha256``.

    A file whose size and modification time are unchanged since it was last hashed is not hashed again.
    """
    relative, expected = row.get(name + "_path", ""), row.get(name + "_sha256", "")
    if not relative or not expected:
        return False
    base = Path(root).resolve()
    path = (base / relative).resolve()
    if base not in path.parents:
        raise ValueError("Manifest path escapes its output root")
    if not path.is_file():
        return False
    stat = path.stat()
    index_path, index = verified_index(base)
    if index.get(relative) == (stat.st_size, stat.st_mtime_ns, expected):
        return True
    if file_hash(path) != expected:
        return False
    index[relative] = (stat.st_size, stat.st_mtime_ns, expected)
    with open(index_path, "a", encoding="utf-8") as stream:
        stream.write(f"{relative}\t{stat.st_size}\t{stat.st_mtime_ns}\t{expected}\n")
    return True


# ---- Worker pools ----

@contextmanager
def worker_pools(log):
    """Own process pools; terminate outstanding workers on interruption or a program error."""
    pools = []
    try:
        yield pools
    except KeyboardInterrupt:
        log.warning("interrupted: terminated %d worker processes; rerun to resume", terminate_children())
        raise SystemExit(130) from None
    except BaseException:
        terminate_children()
        raise
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)


def terminate_children() -> int:
    """Terminate active multiprocessing children (their whole process tree on Windows); return their number."""
    children = multiprocessing.active_children()
    for child in children:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(child.pid)], capture_output=True, check=False)
        else:
            child.terminate()
    for child in children:
        child.join(timeout=5)
    return len(children)
