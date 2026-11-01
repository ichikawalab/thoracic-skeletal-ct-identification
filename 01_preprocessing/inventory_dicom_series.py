"""Stage 1a: inspect every DICOM slice under <root>/<pid>/<study>/<series> and publish the series inventory.

Output: --out (one row per series with headers, geometry, all-slice validation verdict and reason), a private
SQLite index beside it (frozen file listing and every observed Study/Series/SOP UID for the global conflict
audit), plus .log and .meta.json. Reruns resume: series already inspected are kept unless their verdict was
``incomplete``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from common.dicom_series import (
    INVENTORY_FIELDS as FIELDS,
    SUMMARY_TAGS,
    audit_series,
    dicom_files,
    dicom_value,
    open_inventory,
    save_series,
    uid_conflicts,
)
from common.provenance import mark_completion, output_lock, start_run, write_csv_atomic


def scan_series(pid: str, study: str, series_dir: str) -> tuple[dict, list]:
    files = dicom_files(series_dir)
    headers, _, audit = audit_series(files, pid)
    row = dict.fromkeys(FIELDS, "")
    row.update(pid=pid, study_folder=study, series_folder=Path(series_dir).name, n_files=len(files), **audit)
    row["local_validation_status"] = audit["validation_status"]
    if headers:
        row.update({tag: dicom_value(headers[0], tag) for tag in SUMMARY_TAGS})
        row["TransferSyntaxUID"] = ";".join(sorted({str(h.file_meta.get("TransferSyntaxUID", "")) for h in headers}))
    return row, headers


def list_subdirs(path: str) -> list[os.DirEntry]:
    with os.scandir(path) as entries:
        return sorted((e for e in entries if e.is_dir()), key=lambda e: e.name)


def series_jobs(root: str, pid: str) -> list[tuple[str, str, str]]:
    return [(pid, study.name, series.path) for study in list_subdirs(os.path.join(root, pid))
            for series in list_subdirs(study.path)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="DICOM root: <root>/<pid>/<study>/<series>/*.dcm")
    ap.add_argument("--out", required=True, help="inventory CSV; .sqlite, .log and .meta.json are written beside it")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="inspect only the first N subjects (trial run)")
    args = ap.parse_args()
    if args.workers < 1 or args.limit < 0:
        ap.error("workers must be positive and limit nonnegative")
    with output_lock(args.out):
        run_inventory(args)


def run_inventory(args) -> None:
    subjects = [e.name for e in list_subdirs(args.root)]
    log = start_run(__file__, args.out, args, {"columns": FIELDS, "subjects_total": len(subjects)})
    jobs = []
    for index, pid in enumerate(subjects, 1):
        jobs.extend(series_jobs(args.root, pid))
        if index % 1000 == 0 or index == len(subjects):
            log.info("enumerated %d/%d subjects (%d series)", index, len(subjects), len(jobs))
    db = open_inventory(args.out + ".sqlite", args.root, jobs, subjects, log=log)
    try:
        todo = {r[0] for r in db.execute(
            "SELECT path FROM series WHERE result IS NULL OR json_extract(result,'$.validation_status')='incomplete'")}
        requested = set(subjects[:args.limit] if args.limit else subjects)
        pending = [job for job in jobs
                   if job[0] in requested and Path(job[2]).relative_to(args.root).as_posix() in todo]
        counts = dict(db.execute("SELECT pid,COUNT(*) FROM series GROUP BY pid"))
        completed, logged = 0, 0
        with ThreadPoolExecutor(args.workers) as pool:
            for _, subject_jobs in itertools.groupby(pending, key=lambda job: job[0]):
                for row, headers in pool.map(lambda job: scan_series(*job), subject_jobs):
                    row["n_series_subject"] = counts[row["pid"]]
                    save_series(db, args.root, row, headers)
                    completed += 1
                if completed - logged >= 500 or completed == len(pending):
                    logged = completed
                    log.info("inspected %d/%d pending series", completed, len(pending))
        complete = not args.limit and db.execute("SELECT COUNT(*) FROM series WHERE result IS NULL").fetchone()[0] == 0
        conflicts = uid_conflicts(db, args.root) if complete else set()
        rows = []
        for rel, result in db.execute("SELECT path,result FROM series WHERE result IS NOT NULL ORDER BY path"):
            row = json.loads(result)
            if rel in conflicts:
                row["validation_status"] = "hold"
                row["validation_reason"] = ";".join(filter(None, [row["validation_reason"], "global_UID_conflict"]))
            rows.append(row)
        write_csv_atomic(pd.DataFrame(rows, columns=FIELDS), args.out)
        studies = json.loads(db.execute("SELECT value FROM metadata WHERE key='studies'").fetchone()[0])
        populated = {(pid, study) for pid, study, _ in jobs}
        mark_completion(args.out, complete, subjects=subjects, empty_subjects=sorted(set(subjects) - set(counts)),
                        empty_studies=[study for study in studies if tuple(study) not in populated],
                        n_series=len(jobs))
        log.info("inventory: %d/%d series, complete=%s", len(rows), len(jobs), complete)
    finally:
        db.close()


if __name__ == "__main__":
    main()
