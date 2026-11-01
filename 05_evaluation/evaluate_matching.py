"""Tables and figure of the evaluation.

Input: the matching directory written by 04_matching/match_examinations.py and the NLST participant table
distributed with the collection on The Cancer Imaging Archive. Output (--out-dir): identification.csv (rank-k
rates and 1:1 verification per method), comparisons.csv, strata.csv, failures.csv, participants.csv,
flowchart.csv and the two-panel figure (TIFF, PDF).
Every rate has all queries as its denominator (intention to match): a query whose own T0 a method cannot score
counts as not identified at any rank and as rejected at every threshold.

Sections
  1. Candidate retrieval   rank_table, retrieval, comparisons, cmc   -> identification, comparisons
  2. 1:1 verification      histogram_counts, verification            -> identification (columns)
  3. Strata                query_pairs, strata                       -> strata
  4. Failure analysis      failures                                  -> failures
  5. Description           participants_table, flowchart             -> participants, flowchart
  6. Figure                figure                                    -> figure_cmc_verification.tiff/.pdf
  7. Driver                run_evaluation, main
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from common.provenance import (
    STRUCTURES,
    file_hash,
    mark_completion,
    output_lock,
    require_completion,
    start_run,
    write_csv_atomic,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# figures are drawn at their printed size with 8 pt text
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.6, "axes.labelsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "xtick.minor.width": 0.4, "xtick.minor.size": 1.5, "pdf.fonttype": 42,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
STYLE = {"integrated": dict(color="#C44E52", linestyle="-", marker="o", label="Integrated"),
         "vertebrae": dict(color="#4C72B0", linestyle="--", marker="s", label="Vertebrae-only"),
         "sternum": dict(color="#55A868", linestyle="-.", marker="^", label="Sternum-only")}
LEGEND_SAMPLE = 3.75  # length of a legend line sample in font sizes (30 pt: five whole dashes of the dashed line)


def method_legend(axis, loc: str) -> None:
    """Legend of the plotted methods; each dash pattern is centred on its marker so that the sample is symmetric."""
    lines, labels = axis.get_legend_handles_labels()
    patterns = {"--": plt.rcParams["lines.dashed_pattern"], "-.": plt.rcParams["lines.dashdot_pattern"]}
    samples = []
    for line in lines:
        sample = matplotlib.lines.Line2D([], [])
        sample.update_from(line)
        pattern = patterns.get(line.get_linestyle())
        if pattern:  # start inside the pattern so that the middle of a dash falls on the middle of the sample
            length = LEGEND_SAMPLE * plt.rcParams["legend.fontsize"] / line.get_linewidth()  # in dash units
            sample.set_linestyle(((pattern[0] - length) / 2 % sum(pattern), pattern))
        samples.append(sample)
    axis.legend(samples, labels, loc=loc, frameon=True, handlelength=LEGEND_SAMPLE)

METHODS = ["sternum", "vertebrae", "integrated"]
COMPARISONS = [("integrated", "sternum"), ("integrated", "vertebrae"), ("vertebrae", "sternum")]
RANKS = (1, 5, 10)
CMC_MAX_RANK = 10
FAR_TARGETS = (0.001, 0.01)
COMMON_STRATA = {"13": lambda n: n == 13, "12": lambda n: n == 12, "11": lambda n: n == 11, "<=10": lambda n: n <= 10}
# Descriptive strata from the NLST participant table (codes: gender 1 male, 2 female; race 1 White,
# 2 Black, 3 Asian, other codes pooled; cigsmok 1 current, 0 former smoker; age at randomization, with the
# youngest stratum open below so that records outside the nominal 55-74 range are kept).
DEMOGRAPHIC_STRATA = {
    "sex_male": lambda p: p["gender"].eq("1"), "sex_female": lambda p: p["gender"].eq("2"),
    "age_lt60": lambda p: p["age"] < 60, "age_60_64": lambda p: p["age"].between(60, 64),
    "age_65_74": lambda p: p["age"].between(65, 74),
    "race_white": lambda p: p["race"].eq("1"), "race_black": lambda p: p["race"].eq("2"),
    "race_asian": lambda p: p["race"].eq("3"), "race_other_or_unknown": lambda p: ~p["race"].isin(("1", "2", "3")),
    "smoking_current": lambda p: p["cigsmok"].eq("1"), "smoking_former": lambda p: p["cigsmok"].eq("0"),
}
PARTICIPANT_COLUMNS = ["pid", "age", "gender", "race", "cigsmok", "scr_days0", "scr_days2"]
RACE_LEVELS = (("1", "white"), ("2", "black"), ("3", "asian"))  # every other code: other_or_unknown
Z = 1.959964  # two-sided 95%
BOOTSTRAP, SEED = 2000, 0


def wilson(k: int, n: int) -> tuple:
    """Wilson score interval of a proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p, d = k / n, 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / d
    half = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / d
    return (centre - half, centre + half)


def holm(p: list) -> list:
    """Holm step-down adjustment of p-values."""
    order = np.argsort(p)
    adjusted, running = np.empty(len(p)), 0.0
    for i, j in enumerate(order):
        running = max(running, (len(p) - i) * p[j])
        adjusted[j] = min(1.0, running)
    return adjusted.tolist()


def far_label(target: float) -> str:
    return f"far{target * 100:g}pct"


# ---- 1. Candidate retrieval: rank of the own T0 among all gallery examinations ----

def rank_table(results: pd.DataFrame) -> pd.DataFrame:
    """True rank per query (rows) and method (columns); NaN when the method could not score the query's own T0."""
    return results.pivot(index="pid", columns="method", values="true_rank")[METHODS]


def retrieval(ranks: pd.DataFrame) -> pd.DataFrame:
    """Rank-k identification rates with Wilson CIs per method; the primary endpoint is rank 1 of ``integrated``."""
    rows = []
    for method in METHODS:
        row = {"method": method, "n_queries": len(ranks), "n_scored": int(ranks[method].notna().sum())}
        for k in RANKS:
            hits = int((ranks[method] <= k).sum())
            row.update({f"rank{k}_hits": hits, f"rank{k}_rate": hits / len(ranks)})
            row[f"rank{k}_ci_low"], row[f"rank{k}_ci_high"] = wilson(hits, len(ranks))
        rows.append(row)
    return pd.DataFrame(rows)


def comparisons(ranks: pd.DataFrame) -> pd.DataFrame:
    """Paired rank-1 comparison: 2 x 2 table, exact McNemar (two-sided binomial test on the discordant queries),
    Holm across the three comparisons, and a percentile bootstrap CI of the difference in percentage points."""
    rng = np.random.default_rng(SEED)
    hit = (ranks <= 1).to_numpy()
    index = {method: i for i, method in enumerate(METHODS)}
    rows = []
    for a, b in COMPARISONS:
        ha, hb = hit[:, index[a]], hit[:, index[b]]
        only_a, only_b = int((ha & ~hb).sum()), int((~ha & hb).sum())
        p = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        draws = rng.integers(0, len(ha), size=(BOOTSTRAP, len(ha)))
        difference = (ha[draws].mean(axis=1) - hb[draws].mean(axis=1)) * 100
        rows.append({"method_a": a, "method_b": b, "both_hit": int((ha & hb).sum()), "only_a_hit": only_a,
                     "only_b_hit": only_b, "neither_hit": int((~ha & ~hb).sum()), "p_mcnemar": p,
                     "difference_pp": (ha.mean() - hb.mean()) * 100,
                     "difference_ci_low": np.percentile(difference, 2.5),
                     "difference_ci_high": np.percentile(difference, 97.5)})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p_mcnemar"].tolist())
    return table


def cmc(ranks: pd.DataFrame) -> pd.DataFrame:
    """Cumulative match characteristic: fraction of queries whose own T0 is within the top k."""
    ks = np.arange(1, CMC_MAX_RANK + 1)
    return pd.DataFrame({"rank": ks, **{m: [(ranks[m] <= k).mean() for k in ks] for m in METHODS}})


# ---- 2. 1:1 verification: genuine scores against the histogram of all impostor scores ----

def histogram_counts(table: pd.DataFrame, n_bins: int) -> dict:
    """{method: dense bin counts} from the sparse impostor histogram written by the matching stage.

    A method without any scored impostor pair (no shared structure in any pair) keeps an all-zero histogram.
    """
    counts = {method: np.zeros(n_bins, dtype=np.int64) for method in METHODS}
    for method, part in table.groupby("method"):
        dense = np.zeros(n_bins, dtype=np.int64)
        dense[part["bin"].to_numpy(dtype=int)] = part["count"].to_numpy(dtype=np.int64)
        counts[method] = dense
    return counts


def verification(genuine: np.ndarray, counts: np.ndarray, n_impostor_pairs: int) -> tuple:
    """AUC, EER and FRR at the FAR targets; thresholds are the histogram edges (operating points, not selections).

    ``n_impostor_pairs`` is every query-gallery pair other than the own T0; pairs without a score (no shared
    structure) are in the denominator and are never accepted, like an unscorable genuine pair. A genuine score
    inside a bin is placed by linear interpolation for the AUC. A FAR target no edge attains gives NaN.
    """
    n_bins, unscored = len(counts), int(n_impostor_pairs - counts.sum())
    edges = np.linspace(0, 1, n_bins + 1)[:-1]
    far = np.cumsum(counts[::-1])[::-1] / n_impostor_pairs  # impostor score >= edge
    scored = np.sort(genuine[np.isfinite(genuine)])
    accepted = len(scored) - np.searchsorted(scored, edges, side="left")  # genuine score >= edge
    frr = 1 - accepted / len(genuine)
    below = unscored + np.concatenate([[0], np.cumsum(counts)])
    position = np.clip(scored * n_bins, 0, n_bins - 1e-9)
    bin_index = position.astype(int)
    auc = ((below[bin_index] + counts[bin_index] * (position - bin_index)) / n_impostor_pairs).sum() / len(genuine)
    j = int(np.argmin(np.abs(far - frr)))
    summary = {"n_impostor_pairs": n_impostor_pairs, "n_impostor_unscored": unscored, "auc": auc,
               "eer": (far[j] + frr[j]) / 2}
    for target in FAR_TARGETS:
        attained = np.flatnonzero(far <= target)
        label = far_label(target)
        keys = (f"threshold_{label}", f"far_{label}", f"frr_{label}", f"frr_{label}_ci_low", f"frr_{label}_ci_high")
        values = [np.nan] * 5
        if attained.size:
            j = attained[0]
            values = [edges[j], far[j], frr[j], *wilson(len(genuine) - accepted[j], len(genuine))]
        summary.update(zip(keys, values, strict=True))
    return summary, pd.DataFrame({"threshold": edges, "far": far, "tpr": 1 - frr})


# ---- 3. Strata: scanner model and number of structures shared with the own T0 ----

def query_pairs(cohort: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """One row per query: scanner model agreement between T0 and T2 and the structures shared with the own T0."""
    queries = results.loc[results["method"].eq("integrated"), ["pid", "n_common", "true_rank", "true_score"]]
    model = cohort.pivot(index="pid", columns="timepoint", values="ManufacturerModelName")
    pairs = queries.set_index("pid")
    pairs["same_model"] = model.loc[pairs.index, "T0"].eq(model.loc[pairs.index, "T2"]).to_numpy()
    return pairs


def strata(ranks: pd.DataFrame, pairs: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    """Rank-1 rates per stratum and method; one test (integrated, same vs different model), other strata descriptive."""
    groups = {"same_model": pairs["same_model"], "different_model": ~pairs["same_model"]}
    groups.update({f"common_{label}": rule(pairs["n_common"]) for label, rule in COMMON_STRATA.items()})
    persons = participants.set_index("pid").reindex(ranks.index)
    persons["age"] = pd.to_numeric(persons["age"])
    groups.update({label: rule(persons) for label, rule in DEMOGRAPHIC_STRATA.items()})
    rows = []
    for label, mask in groups.items():
        mask = mask.reindex(ranks.index).to_numpy()
        n = int(mask.sum())
        for method in METHODS:
            hits = int((ranks.loc[mask, method] <= 1).sum())
            low, high = wilson(hits, n)
            rows.append({"stratum": label, "method": method, "n": n, "rank1_hits": hits,
                         "rank1_rate": hits / n if n else np.nan, "ci_low": low, "ci_high": high})
    table = pd.DataFrame(rows)
    same = pairs["same_model"].reindex(ranks.index).to_numpy()
    hit = (ranks["integrated"] <= 1).to_numpy()
    contingency = np.array([[(hit & same).sum(), (~hit & same).sum()], [(hit & ~same).sum(), (~hit & ~same).sum()]])
    testable = contingency.sum(axis=0).all() and contingency.sum(axis=1).all()  # an empty margin has no test
    table["p_chi2_integrated_model"] = stats.chi2_contingency(contingency, correction=False)[1] if testable else np.nan
    return table


# ---- 4. Failure analysis: queries the integrated method did not rank first (descriptive) ----

def failures(ranks: pd.DataFrame, pairs: pd.DataFrame, structures: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Per failed query: shared structures, scanner change, per-structure similarity to the own T0 and to the
    rank-1 impostor, and whether renumbering the query's vertebrae by one level would have resolved it."""
    failed = ranks[~(ranks["integrated"] <= 1)].copy()
    failed.columns = [f"rank_{m}" for m in METHODS]
    detail = structures.set_index("pid").loc[failed.index]
    mate = detail[[f"mate_{s}" for s in STRUCTURES]]
    impostor = detail[[f"impostor_{s}" for s in STRUCTURES]]
    vertebrae_top1 = results[results["method"].eq("vertebrae")].set_index("pid").loc[failed.index, "top1_score"]
    failed["n_common"] = pairs.loc[failed.index, "n_common"]
    failed["same_model"] = pairs.loc[failed.index, "same_model"]
    failed["T1_common"] = mate["mate_vertebrae_T1"].notna()
    failed["T12_common"] = mate["mate_vertebrae_T12"].notna()
    failed["mate_score"] = mate.mean(axis=1)
    failed["impostor_pid"] = detail["impostor_pid"]
    failed["impostor_score"] = impostor.mean(axis=1)
    failed["mate_min_structure"] = mate.min(axis=1)
    failed["n_common_impostor"] = impostor.notna().sum(axis=1)
    failed["n_structures_impostor_higher"] = (impostor.to_numpy() > mate.to_numpy()).sum(axis=1)
    shifted = detail[["vertebrae_shift_up", "vertebrae_shift_down"]].max(axis=1)
    failed["vertebrae_shift_score"] = shifted
    failed["shift_resolves"] = shifted > vertebrae_top1
    return failed.reset_index()


# ---- 5. Description: participants, acquisition and flowchart counts ----

def participants_table(cohort: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    """Long table (group, variable, level, value) for the gallery and the queries."""
    persons = participants.set_index("pid")
    described = set(cohort.loc[cohort["role"].isin(("gallery", "query")), "pid"])
    if not persons.index.is_unique or not described <= set(persons.index):
        raise ValueError("Participant table must list every gallery and query pid exactly once")
    rows = []
    for role in ("gallery", "query"):
        exams = cohort[cohort["role"].eq(role)]
        age = pd.to_numeric(persons.loc[exams["pid"], "age"])
        female = pd.to_numeric(persons.loc[exams["pid"], "gender"]).eq(2)  # NLST coding: 1 male, 2 female
        race = persons.loc[exams["pid"], "race"]
        thickness = exams["thickness_mm"].astype(float)
        rows += [(role, "examinations", "", len(exams)),
                 (role, "age_at_randomization", "median", age.median()),
                 (role, "age_at_randomization", "q1", age.quantile(0.25)),
                 (role, "age_at_randomization", "q3", age.quantile(0.75)),
                 (role, "female", "n", int(female.sum())),
                 *[(role, "race", label, int(race.eq(code).sum())) for code, label in RACE_LEVELS],
                 (role, "race", "other_or_unknown", int((~race.isin([c for c, _ in RACE_LEVELS])).sum())),
                 (role, "slice_thickness_mm", "median", thickness.median()),
                 (role, "slice_thickness_mm", "q1", thickness.quantile(0.25)),
                 (role, "slice_thickness_mm", "q3", thickness.quantile(0.75))]
        for column in ("manufacturer_group", "kernel_class", "n_usable"):
            rows += [(role, column, str(level), int(n)) for level, n in exams[column].value_counts().items()]
    # Days between the T0 and T2 examinations of the queries (scr_days: days from randomization to each screening).
    queries = cohort.loc[cohort["role"].eq("query"), "pid"]
    days = persons.loc[queries, ["scr_days0", "scr_days2"]].apply(pd.to_numeric, errors="coerce")
    interval = days["scr_days2"] - days["scr_days0"]
    rows += [("query", "interval_days", "median", interval.median()),
             ("query", "interval_days", "q1", interval.quantile(0.25)),
             ("query", "interval_days", "q3", interval.quantile(0.75)),
             ("query", "interval_days", "missing", int(interval.isna().sum()))]
    return pd.DataFrame(rows, columns=["group", "variable", "level", "value"])


def flowchart(cohort: pd.DataFrame, retrieval_table: pd.DataFrame) -> pd.DataFrame:
    roles = cohort["role"].value_counts()
    single = cohort[cohort["role"].eq("single_structure")]["timepoint"].value_counts()
    rows = [("subjects_with_eligible_examination", cohort["pid"].nunique()),
            ("eligible_T0", int(cohort["timepoint"].eq("T0").sum())),
            ("eligible_T2", int(cohort["timepoint"].eq("T2").sum())),
            ("T0_single_structure", int(single.get("T0", 0))), ("T2_single_structure", int(single.get("T2", 0))),
            ("gallery", int(roles["gallery"])), ("queries", int(roles["query"])),
            ("gallery_without_query", int(roles["gallery"] - roles["query"])),
            ("T2_without_reference", int(roles.get("no_reference", 0)))]
    rows += [(f"queries_scored_{r.method}", int(r.n_scored)) for r in retrieval_table.itertuples()]
    return pd.DataFrame(rows, columns=["level", "n"])


# ---- 6. Figure: CMC and verification ROC ----

def figure(cmc_table: pd.DataFrame, curves: dict, path: str) -> None:
    """Panel A: cumulative match characteristic to rank 10. Panel B: verification ROC with the FAR operating points."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(174 / 25.4, 76 / 25.4))
    for method in METHODS:
        style = STYLE[method]
        left.plot(cmc_table["rank"], cmc_table[method] * 100, linewidth=1.2, markersize=3.5, **style)
        curve = curves[method]
        right.plot(np.where(curve["far"] > 0, curve["far"], np.nan) * 100, curve["tpr"] * 100, linewidth=1.2,
                   **{k: v for k, v in style.items() if k != "marker"})
    left.set(xlabel="Rank", ylabel="Cumulative identification rate (%)", xticks=cmc_table["rank"], ylim=(0, 100))
    right.set(xscale="log", xlabel="False acceptance rate (%)", ylabel="True acceptance rate (%)",
              xlim=(1e-4, 100), ylim=(0, 100))
    for target in FAR_TARGETS:
        right.axvline(target * 100, color="#A8A8A8", linewidth=0.5, linestyle=":", zorder=0)
    for axis, label in ((left, "a"), (right, "b")):  # panel letters in lower case
        axis.grid(True, axis="y", color="#D0D0D0", linewidth=0.4, alpha=0.65)
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontsize=9, fontweight="bold")
    method_legend(left, "lower right")  # one legend: the methods are the same in b
    fig.tight_layout()
    fig.savefig(path + ".tiff", dpi=1200, bbox_inches="tight", pad_inches=0.02,  # line art: 1200 dpi
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


# ---- 7. Driver ----

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matching-dir", required=True, help="directory written by the matching stage")
    ap.add_argument("--participants", required=True,
                    help="NLST participant table (nlst_780_prsn_*.csv) with pid, age, gender, race, cigsmok, "
                         "scr_days0 and scr_days2")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    with output_lock(os.path.join(args.out_dir, "identification.csv")):
        run_evaluation(args)


def run_evaluation(args: argparse.Namespace) -> None:
    results_path = os.path.join(args.matching_dir, "matching_results.csv")
    matching_run = require_completion(results_path)
    anchor = os.path.join(args.out_dir, "identification.csv")
    log = start_run(__file__, anchor, args, {"bootstrap": BOOTSTRAP, "seed": SEED, "far_targets": FAR_TARGETS,
                                              "cmc_max_rank": CMC_MAX_RANK})
    for name, digest in matching_run["files"].items():  # the completion record covers the side tables too
        if file_hash(os.path.join(args.matching_dir, name)) != digest:
            raise ValueError(f"{name} changed since the matching run")
    read = dict(dtype={"pid": str, "top1_pid": str, "impostor_pid": str})
    results = pd.read_csv(results_path, **read)
    load = {name: pd.read_csv(os.path.join(args.matching_dir, name + ".csv"), **read)
            for name in ("query_structures", "impostor_histogram")}
    load["cohort"] = pd.read_csv(os.path.join(args.matching_dir, "cohort.csv"), dtype=str, keep_default_na=False)
    participants = pd.read_csv(args.participants, dtype=str, usecols=PARTICIPANT_COLUMNS)
    ranks = rank_table(results)
    retrieval_table = retrieval(ranks)
    counts = histogram_counts(load["impostor_histogram"], matching_run["run_identity"]["extra"]["n_bins"])
    n_impostor_pairs = matching_run["n_query"] * (matching_run["n_gallery"] - 1)
    curves, summaries = {}, []
    for method in METHODS:
        genuine = results.loc[results["method"].eq(method)].set_index("pid").loc[ranks.index, "true_score"]
        summary, curves[method] = verification(genuine.to_numpy(dtype=float), counts[method], n_impostor_pairs)
        threshold = summary[f"threshold_{far_label(FAR_TARGETS[0])}"]
        summary[f"rank1_and_accepted_{far_label(FAR_TARGETS[0])}"] = (
            ((ranks[method] <= 1) & (genuine.to_numpy() >= threshold)).mean() if np.isfinite(threshold) else np.nan)
        summaries.append(summary)
    retrieval_table = retrieval_table.join(pd.DataFrame(summaries))
    pairs = query_pairs(load["cohort"], results)
    tables = {"comparisons": comparisons(ranks), "strata": strata(ranks, pairs, participants),
              "failures": failures(ranks, pairs, load["query_structures"], results),
              "participants": participants_table(load["cohort"], participants),
              "flowchart": flowchart(load["cohort"], retrieval_table)}
    files = {}
    for name, table in tables.items():
        path = os.path.join(args.out_dir, name + ".csv")
        write_csv_atomic(table, path)
        files[name + ".csv"] = file_hash(path)
    stem = os.path.join(args.out_dir, "figure_cmc_verification")
    figure(cmc(ranks), curves, stem)
    files.update({f"figure_cmc_verification{s}": file_hash(stem + s) for s in (".tiff", ".pdf")})
    write_csv_atomic(retrieval_table, anchor)
    log.info("rank-1: %s", json.dumps(retrieval_table.set_index("method")["rank1_rate"].round(4).to_dict()))
    log.info("ms per query in matching: %s", matching_run.get("ms_per_query"))
    mark_completion(anchor, True, files=files)


if __name__ == "__main__":
    main()
