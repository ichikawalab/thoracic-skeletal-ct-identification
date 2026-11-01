"""Sensitivity analysis: identification rate as a function of gallery size.

A fixed random set of queries is matched against galleries that always contain their own T0 and grow by random
distractors to each size (nested within a repeat); the whole procedure is repeated with new random query sets.
Scoring is the main analysis unchanged (04_matching/match_examinations.py, gallery statistics recomputed per
gallery). Output (--out-dir): gallery_size.csv (size x repeat x method) and gallery_size.tiff/.pdf.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

from common.provenance import file_hash, mark_completion, output_lock, require_completion, start_run, write_csv_atomic

HERE = Path(__file__).resolve().parent


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


match = load("match_examinations", "04_matching/match_examinations.py")
evaluate = load("evaluate_matching", "05_evaluation/evaluate_matching.py")
plt, METHODS, STYLE = evaluate.plt, evaluate.METHODS, evaluate.STYLE
SIZES = (1000, 2000, 5000, 10000)  # plus the full gallery
N_QUERIES, REPEATS = 1000, 20


def sensitivity(features: pd.DataFrame, cohort: pd.DataFrame, sizes, n_queries: int, repeats: int) -> pd.DataFrame:
    """Rank-1 and rank-5 rates per gallery size, repeat and method for fixed random query sets."""
    gallery_pids = cohort.loc[cohort["role"].eq("gallery"), "pid"].to_numpy()
    query_pids = cohort.loc[cohort["role"].eq("query"), "pid"].to_numpy()
    if not 0 < n_queries <= len(query_pids):
        raise ValueError(f"n_queries must be between 1 and the number of queries ({len(query_pids)})")
    sizes = sorted({size for size in sizes if n_queries <= size <= len(gallery_pids)} | {len(gallery_pids)})
    rows = []
    for repeat in range(repeats):
        rng = np.random.default_rng(repeat)
        queries = rng.choice(query_pids, n_queries, replace=False)
        distractors = rng.permutation(np.setdiff1d(gallery_pids, queries))
        for size in sizes:
            gallery = set(queries) | set(distractors[:size - n_queries])
            subset = cohort[(cohort["role"].eq("gallery") & cohort["pid"].isin(gallery))
                            | (cohort["role"].eq("query") & cohort["pid"].isin(queries))]
            results = match.match(features, subset)["matching_results"]
            for method, part in results.groupby("method"):
                rows.append({"gallery_size": size, "repeat": repeat, "method": method, "n_queries": n_queries,
                             "rank1_rate": (part["true_rank"] <= 1).mean(),
                             "rank5_rate": (part["true_rank"] <= 5).mean()})
    return pd.DataFrame(rows)


def figure(table: pd.DataFrame, path: str) -> None:
    summary = table.groupby(["method", "gallery_size"])["rank1_rate"].agg(["mean", "min", "max"]).reset_index()
    fig, ax = plt.subplots(figsize=(84 / 25.4, 70 / 25.4))  # single column
    for method in METHODS:
        part = summary[summary["method"].eq(method)]
        style = STYLE[method]
        ax.plot(part["gallery_size"], part["mean"] * 100, linewidth=1.2, markersize=3.5, **style)
        ax.fill_between(part["gallery_size"], part["min"] * 100, part["max"] * 100, color=style["color"], alpha=0.25,
                        linewidth=0)
    sizes = sorted(summary["gallery_size"].unique())
    ax.set(xscale="log", xlabel="Gallery size (examinations)", ylabel="Rank-1 identification rate (%)", ylim=(0, 100),
           xticks=sizes, xticklabels=[f"{size:,}" for size in sizes])  # the evaluated sizes only
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(True, axis="y", color="#D0D0D0", linewidth=0.4, alpha=0.65)
    evaluate.method_legend(ax, "lower left")
    fig.tight_layout()
    fig.savefig(path + ".tiff", dpi=1200, bbox_inches="tight", pad_inches=0.02,  # line art: 1200 dpi
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matching-dir", required=True, help="directory written by the matching stage (cohort.csv)")
    ap.add_argument("--features", required=True, help="the shape-feature table the matching stage used")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    with output_lock(os.path.join(args.out_dir, "gallery_size.csv")):
        run(args)


def run(args: argparse.Namespace) -> None:
    matching_run = require_completion(os.path.join(args.matching_dir, "matching_results.csv"))
    if matching_run["run_identity"]["inputs"].get("features") != file_hash(args.features):
        raise ValueError("Feature table differs from the one the matching stage used")
    cohort_path = os.path.join(args.matching_dir, "cohort.csv")
    if matching_run["files"].get("cohort.csv") != file_hash(cohort_path):
        raise ValueError("cohort.csv changed since the matching run")
    anchor = os.path.join(args.out_dir, "gallery_size.csv")
    log = start_run(__file__, anchor, args, {"sizes": SIZES, "n_queries": N_QUERIES, "repeats": REPEATS})
    cohort = pd.read_csv(cohort_path, dtype=str, keep_default_na=False)
    features = pd.read_csv(args.features, dtype=str, keep_default_na=False)
    table = sensitivity(features, cohort, SIZES, N_QUERIES, REPEATS)
    stem = os.path.join(args.out_dir, "gallery_size")
    figure(table, stem)
    write_csv_atomic(table, anchor)
    summary = table.groupby(["gallery_size", "method"])["rank1_rate"].mean().unstack()[METHODS]
    log.info("mean rank-1 by gallery size:\n%s", summary.round(4).to_string())
    mark_completion(anchor, True, files={f"gallery_size{s}": file_hash(stem + s) for s in (".tiff", ".pdf")})


if __name__ == "__main__":
    main()
