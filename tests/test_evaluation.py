"""Confidence intervals, paired comparisons, verification from the impostor histogram, strata, failures; both stages."""
import argparse
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from common.provenance import mark_completion, start_run
from tests.conftest import load_stage_script
from tests.test_matching import STRUCTURES, exam_features

evaluate = load_stage_script("05_evaluation/evaluate_matching.py")
match = load_stage_script("04_matching/match_examinations.py")
gallery_size = load_stage_script("05_evaluation/gallery_size.py")
METHODS = evaluate.METHODS


def test_wilson_interval_and_holm_adjustment():
    low, high = evaluate.wilson(50, 100)
    assert (low, high) == pytest.approx((0.4038, 0.5962), abs=1e-4)
    assert evaluate.wilson(0, 10)[0] == pytest.approx(0) and evaluate.wilson(10, 10)[1] == pytest.approx(1)
    assert np.isnan(evaluate.wilson(0, 0)).all()
    assert evaluate.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_rank_rates_and_cmc_at_the_rank_boundaries():
    ranks = pd.DataFrame({m: [1, 5, 6, 10, 11, 100, 101, np.nan] for m in METHODS}, index=list("abcdefgh"))
    table = evaluate.retrieval(ranks).set_index("method")
    counts = table.loc["integrated", ["n_queries", "n_scored", "rank1_hits", "rank5_hits", "rank10_hits"]]
    assert counts.tolist() == [8, 7, 1, 2, 4]
    assert table.loc["integrated", ["rank1_rate", "rank5_rate", "rank10_rate"]].tolist() == [1 / 8, 2 / 8, 4 / 8]
    assert table.loc["integrated", ["rank1_ci_low", "rank1_ci_high"]].tolist() == pytest.approx(evaluate.wilson(1, 8))
    curve = evaluate.cmc(ranks)
    assert len(curve) == 10 and curve["integrated"].iloc[0] == 1 / 8 and curve["integrated"].iloc[-1] == 4 / 8


def test_paired_comparison_uses_exact_mcnemar_holm_and_a_seeded_paired_bootstrap():
    rows = []
    for i in range(20):  # integrated hits 0-15, sternum hits 0-9 and 16-17, vertebrae never scored
        rows += [{"pid": str(i), "method": "integrated", "true_rank": 1.0 if i < 16 else 2.0},
                 {"pid": str(i), "method": "sternum", "true_rank": 1.0 if i < 10 or i in (16, 17) else 3.0},
                 {"pid": str(i), "method": "vertebrae", "true_rank": np.nan}]
    ranks = evaluate.rank_table(pd.DataFrame(rows))
    comparison = evaluate.comparisons(ranks).set_index(["method_a", "method_b"])
    row = comparison.loc[("integrated", "sternum")]
    assert row[["both_hit", "only_a_hit", "only_b_hit", "neither_hit"]].tolist() == [10, 6, 2, 2]
    assert row["p_mcnemar"] == pytest.approx(74 / 256) and row["difference_pp"] == pytest.approx(20)
    assert row["difference_ci_low"] < 20 < row["difference_ci_high"]
    p = comparison["p_mcnemar"]
    assert p[("integrated", "vertebrae")] == pytest.approx(2 * 0.5**16)
    assert p[("vertebrae", "sternum")] == pytest.approx(2 * 0.5**12)
    assert comparison["p_holm"].tolist() == pytest.approx([74 / 256, 3 * 2 * 0.5**16, 2 * 2 * 0.5**12])
    again = evaluate.comparisons(ranks).set_index(["method_a", "method_b"])
    assert again["difference_ci_low"].tolist() == comparison["difference_ci_low"].tolist()  # seeded


def test_verification_from_histogram_separates_genuine_from_impostor_scores():
    counts = np.zeros(10, dtype=np.int64)
    counts[5] = 1000  # every impostor score in [0.5, 0.6)
    genuine = np.array([0.85, 0.9, 0.95, np.nan])  # one query whose own T0 could not be scored
    summary, curve = evaluate.verification(genuine, counts, 1000)
    assert summary["auc"] == pytest.approx(0.75) and summary["eer"] == pytest.approx(0.125)
    assert summary["threshold_far0.1pct"] == pytest.approx(0.6) and summary["far_far0.1pct"] == 0
    assert summary["frr_far0.1pct"] == pytest.approx(0.25) and summary["frr_far1pct"] == pytest.approx(0.25)
    assert curve["far"].iloc[0] == 1 and curve["tpr"].iloc[0] == pytest.approx(0.75)
    dense = evaluate.histogram_counts(pd.DataFrame({"method": ["sternum"], "bin": [5], "count": [1000]}), 10)
    assert dense["sternum"].tolist() == counts.tolist()
    summary, curve = evaluate.verification(genuine, counts, 2000)  # 1000 impostor pairs without a score
    assert curve["far"].iloc[0] == 0.5 and summary["auc"] == pytest.approx(0.75)
    summary, _ = evaluate.verification(genuine, np.array([0] * 9 + [1000]), 1000)  # every impostor scores >= 0.9
    assert np.isnan(summary["threshold_far0.1pct"]) and np.isnan(summary["frr_far1pct"])
    counts = np.zeros(10, dtype=np.int64)
    counts[[3, 6, 8]] = [995, 4, 1]  # FAR 0.5% from edge 0.4, 0.1% from edge 0.7
    summary, _ = evaluate.verification(np.array([0.5, 0.95, np.nan]), counts, 1200)
    assert summary["n_impostor_unscored"] == 200 and summary["threshold_far1pct"] == pytest.approx(0.4)
    assert summary["threshold_far0.1pct"] == pytest.approx(0.7) and summary["far_far0.1pct"] == pytest.approx(1 / 1200)
    assert summary["frr_far1pct"] == pytest.approx(1 / 3) and summary["frr_far0.1pct"] == pytest.approx(2 / 3)
    assert (summary["frr_far0.1pct_ci_low"], summary["frr_far0.1pct_ci_high"]) == pytest.approx(evaluate.wilson(2, 3))


def test_strata_and_failures_describe_the_queries_not_ranked_first():
    """Four queries: shared structures 13/12/11/10, models same/same/different/different, integrated ranks 1/2/1/3."""
    models = {"1": ("A", "A"), "2": ("B", "B"), "3": ("A", "B"), "4": ("C", "D")}
    cohort = pd.DataFrame([{"pid": pid, "timepoint": tp, "ManufacturerModelName": models[pid][k], "role": role}
                           for pid in "1234" for k, (tp, role) in enumerate((("T0", "gallery"), ("T2", "query")))])
    n_common = {"1": 13, "2": 12, "3": 11, "4": 10}
    rank = {"integrated": {"1": 1, "2": 2, "3": 1, "4": 3}, "vertebrae": {"1": 1, "2": 3, "3": 2, "4": 1},
            "sternum": {"1": 2, "2": 1, "3": 5, "4": 1}}
    results = pd.DataFrame([{"pid": pid, "method": m, "n_common": n_common[pid] if m != "sternum" else 1,
                             "true_rank": float(rank[m][pid]), "true_score": 0.8, "top1_score": 0.9}
                            for pid in "1234" for m in METHODS])
    ranks = evaluate.rank_table(results)
    pairs = evaluate.query_pairs(cohort, results)
    assert pairs["same_model"].to_dict() == {"1": True, "2": True, "3": False, "4": False}
    assert pairs["n_common"].to_dict() == n_common  # the integrated method's count, shared by every method
    participants = pd.DataFrame({"pid": list("1234"), "age": ["57", "62", "70", "66"], "gender": ["1", "2", "2", "1"],
                                 "race": ["1", "2", "3", "7"], "cigsmok": ["1", "0", "1", "0"]})
    table = evaluate.strata(ranks, pairs, participants).set_index(["stratum", "method"])
    assert table.loc[("sex_female", "integrated"), ["n", "rank1_hits"]].tolist() == [2, 1]
    assert table.loc[("age_65_74", "integrated"), ["n", "rank1_hits"]].tolist() == [2, 1]  # pids 3 and 4
    assert table.loc[("age_lt60", "integrated"), "n"] == 1
    assert table.loc[("race_other_or_unknown", "sternum"), ["n", "rank1_hits"]].tolist() == [1, 1]
    assert table.loc[("smoking_current", "vertebrae"), ["n", "rank1_hits"]].tolist() == [2, 1]
    assert table.loc[("same_model", "integrated"), ["n", "rank1_hits"]].tolist() == [2, 1]
    assert table.loc[("different_model", "sternum"), ["n", "rank1_hits"]].tolist() == [2, 1]
    assert [table.loc[(f"common_{s}", "vertebrae"), "rank1_hits"] for s in ("13", "12", "11", "<=10")] == [1, 0, 0, 1]
    expected_p = stats.chi2_contingency([[1, 1], [1, 1]], correction=False)[1]
    assert table["p_chi2_integrated_model"].iloc[0] == pytest.approx(expected_p)
    mate = {f"mate_{s}": 0.8 for s in evaluate.STRUCTURES}
    shift = {"2": 0.9, "4": 0.95}  # equals / exceeds the vertebrae top-1 score 0.9
    impostor = {f"impostor_{s}": 0.85 if s != "vertebrae_T12" else np.nan for s in evaluate.STRUCTURES}
    structures = pd.DataFrame([{"pid": pid, "impostor_pid": "9", "vertebrae_shift_up": shift.get(pid, 0.5),
                                "vertebrae_shift_down": 0.5, **mate, **impostor,
                                "mate_vertebrae_T1": np.nan if pid == "4" else 0.8} for pid in "1234"])
    failed = evaluate.failures(ranks, pairs, structures, results).set_index("pid")
    assert failed.index.tolist() == ["2", "4"] and failed["rank_integrated"].tolist() == [2, 3]
    assert failed["shift_resolves"].tolist() == [False, True] and failed["T1_common"].tolist() == [True, False]
    assert failed["n_common_impostor"].tolist() == [12, 12]
    assert failed["n_structures_impostor_higher"].tolist() == [12, 11]
    assert failed["same_model"].tolist() == [True, False] and failed["mate_score"].iloc[0] == pytest.approx(0.8)


def test_histogram_counts_keeps_every_method():
    sparse = pd.DataFrame({"method": ["sternum", "integrated"], "bin": [3, 9], "count": [2, 1]})
    counts = evaluate.histogram_counts(sparse, 10)
    assert set(counts) == set(METHODS) and counts["vertebrae"].sum() == 0 and counts["sternum"][3] == 2
    summary, _ = evaluate.verification(np.array([0.5, np.nan]), counts["vertebrae"], n_impostor_pairs=4)
    assert summary["n_impostor_unscored"] == 4 and summary["frr_far0.1pct"] == 0.5


def test_participants_table_and_figure(tmp_path):
    cohort = pd.DataFrame([{"pid": "1", "timepoint": "T0", "role": "gallery", "thickness_mm": 2.5, "n_usable": 13,
                            "manufacturer_group": "GE", "kernel_class": "soft"},
                           {"pid": "1", "timepoint": "T2", "role": "query", "thickness_mm": 1.0, "n_usable": 12,
                            "manufacturer_group": "SIEMENS", "kernel_class": "sharp"}])
    participants = pd.DataFrame({"pid": ["1", "2"], "age": ["60", "70"], "gender": ["2", "1"], "race": ["7", "1"],
                                 "scr_days0": ["12", "9"], "scr_days2": ["740", ""]})
    table = evaluate.participants_table(cohort, participants).set_index(["group", "variable", "level"])["value"]
    assert table[("query", "interval_days", "median")] == 728 and table[("query", "interval_days", "missing")] == 0
    assert table[("gallery", "female", "n")] == 1 and table[("query", "age_at_randomization", "median")] == 60
    assert table[("gallery", "race", "other_or_unknown")] == 1 and table[("query", "race", "white")] == 0
    assert table[("query", "manufacturer_group", "SIEMENS")] == 1 and table[("gallery", "n_usable", "13")] == 1
    ranks = pd.DataFrame({m: [1.0, 2.0, np.nan] for m in METHODS}, index=["1", "2", "3"])
    curve = pd.DataFrame({"threshold": [0, 0.5], "far": [1, 0.001], "tpr": [1, 0.9]})
    evaluate.figure(evaluate.cmc(ranks), dict.fromkeys(METHODS, curve), str(tmp_path / "figure"))
    assert (tmp_path / "figure.pdf").stat().st_size > 0 and (tmp_path / "figure.tiff").stat().st_size > 0


def completed_table(path, rows, script="stage.py", **args):
    start_run(script, str(path), argparse.Namespace(**args))
    pd.DataFrame(rows).to_csv(path, index=False)
    mark_completion(path, True)


def test_both_stages_run_on_completed_inputs_and_record_their_outputs(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    exams = [(pid, tp) for pid in "123456" for tp in ("T0", "T2") if not (pid == "6" and tp == "T2")]
    exams += [("7", "T2"), ("8", "T0")]  # 7: eligible T2 without an eligible T0; 8: only the sternum usable
    qc_rows = [{"pid": pid, "timepoint": tp, "structure": name, "exam_eligible": "True",
                "structure_status": "usable" if pid != "8" or name == "sternum" else "numbering_gap"}
               for pid, tp in exams for name in STRUCTURES]
    completed_table(tmp_path / "qc.csv", qc_rows)
    (tmp_path / "selection").mkdir()
    selection = [{"pid": pid, "timepoint": tp, "status": "selected", "manufacturer_group": "GE",
                  "ManufacturerModelName": "M" if pid < "4" or tp == "T0" else "N", "kernel_class": "soft",
                  "thickness_mm": "2.5"} for pid, tp in exams]
    completed_table(tmp_path / "selection" / "exam_selection.csv", selection)
    (tmp_path / "other").mkdir()
    completed_table(tmp_path / "other" / "exam_selection.csv",
                    [{**row, "ManufacturerModelName": "Z"} for row in selection])
    base = {pid: {name: rng.normal(size=14) for name in STRUCTURES} for pid in "12345678"}
    values = {(pid, tp): base["3"] if (pid, tp) == ("5", "T2") else base[pid] for pid, tp in exams}  # 5 resembles 3
    feature_rows = [row for pid, tp in exams
                    for row in exam_features(pid, tp, values[(pid, tp)], ["sternum"] if pid == "8" else STRUCTURES)]
    (tmp_path / "nifti").mkdir()
    completed_table(tmp_path / "nifti" / "nifti_manifest.csv", [{"pid": "1", "timepoint": "T0"}], "convert_to_nifti.py",
                    selection_dir=str(tmp_path / "selection"))
    (tmp_path / "nifti_other").mkdir()
    completed_table(tmp_path / "nifti_other" / "nifti_manifest.csv", [{"pid": "1", "timepoint": "T2"}],
                    "convert_to_nifti.py", selection_dir=str(tmp_path / "selection"))
    # seg_other: a segmentation the features did not use; seg_stale: segmented from another conversion
    segmentations = (("seg", "T0", "nifti"), ("seg_other", "T2", "nifti"), ("seg_stale", "T0", "nifti_other"))
    for name, timepoint, nifti in segmentations:
        (tmp_path / name).mkdir()
        (tmp_path / name / "label_maps.json").write_text("{}", encoding="utf-8")
        completed_table(tmp_path / name / "segmentation_manifest.csv", [{"pid": "1", "timepoint": timepoint}],
                        "run_segmentation.py", nifti_root=str(tmp_path / nifti))
    completed_table(tmp_path / "features.csv", feature_rows, "extract_shape_features.py", qc=str(tmp_path / "qc.csv"),
                    seg_root=str(tmp_path / "seg"), timepoints="T0,T2")
    pd.DataFrame({"pid": list("1234567"), "age": [60, 61, 62, 63, 64, 65, 66], "gender": [1, 2, 1, 2, 1, 2, 1],
                  "race": 1, "cigsmok": [1, 0, 1, 0, 1, 0, 1], "scr_days0": 10,
                  "scr_days2": [735, 740, 745, 750, None, 760, 765]}).to_csv(tmp_path / "prsn.csv", index=False)
    arguments = dict(features=str(tmp_path / "features.csv"), qc=str(tmp_path / "qc.csv"),
                     nifti_root=str(tmp_path / "nifti"), out_dir=str(tmp_path / "matching"))
    with pytest.raises(ValueError, match="conversion used"):  # a completed selection the conversion did not use
        match.run_matching(argparse.Namespace(**arguments, selection_dir=str(tmp_path / "other"),
                                              seg_root=str(tmp_path / "seg")))
    with pytest.raises(ValueError, match="different segmentation manifest"):
        match.run_matching(argparse.Namespace(**arguments, selection_dir=str(tmp_path / "selection"),
                                              seg_root=str(tmp_path / "seg_other")))
    with pytest.raises(ValueError, match="different NIfTI manifest"):  # seg_stale has the same content as seg
        match.run_matching(argparse.Namespace(**arguments, selection_dir=str(tmp_path / "selection"),
                                              seg_root=str(tmp_path / "seg_stale")))
    match.run_matching(argparse.Namespace(**arguments, selection_dir=str(tmp_path / "selection"),
                                          seg_root=str(tmp_path / "seg")))
    matched = json.loads((tmp_path / "matching" / "matching_results.csv.meta.json").read_text())[-1]
    assert matched["complete"] and (matched["n_gallery"], matched["n_query"], matched["n_no_reference"]) == (6, 5, 1)
    assert matched["n_single_structure"] == 1 and "scaling.csv" not in matched["files"]
    results = pd.read_csv(tmp_path / "matching" / "matching_results.csv", dtype={"pid": str, "top1_pid": str})
    ranks = results.pivot(index="pid", columns="method", values="true_rank")
    assert ranks.loc[["1", "2", "3", "4"], "integrated"].eq(1).all() and ranks.loc["5", "integrated"] > 1
    evaluate.run_evaluation(argparse.Namespace(matching_dir=str(tmp_path / "matching"),
                                               participants=str(tmp_path / "prsn.csv"),
                                               out_dir=str(tmp_path / "evaluation")))
    meta = json.loads((tmp_path / "evaluation" / "identification.csv.meta.json").read_text())[-1]
    assert meta["complete"] and {"figure_cmc_verification.pdf", "figure_cmc_verification.tiff"} <= set(meta["files"])
    assert not any(name.startswith(("cmc", "verification_curve")) for name in meta["files"])
    table = pd.read_csv(tmp_path / "evaluation" / "identification.csv").set_index("method")
    assert table.loc["integrated", "rank1_rate"] == 0.8 and table.loc["integrated", "n_scored"] == 5
    assert np.isnan(table.loc["integrated", "frr_far0.1pct"])  # query 5 scores 1.0 against gallery 3: FAR >= 1/25
    assert np.isnan(table.loc["integrated", "rank1_and_accepted_far0.1pct"])
    assert table.loc["integrated", ["n_impostor_pairs", "n_impostor_unscored", "auc"]].tolist()[:2] == [25, 0]
    assert table.loc["integrated", "auc"] > 0.5 and matched["seconds_matching"] >= 0 and matched["ms_per_query"] >= 0
    flow = pd.read_csv(tmp_path / "evaluation" / "flowchart.csv").set_index("level")["n"]
    assert flow[["gallery", "queries", "gallery_without_query", "T2_without_reference"]].tolist() == [6, 5, 1, 1]
    assert flow[["eligible_T0", "T0_single_structure", "T2_single_structure"]].tolist() == [7, 1, 0]
    failed = pd.read_csv(tmp_path / "evaluation" / "failures.csv", dtype={"pid": str, "impostor_pid": str})
    assert failed["pid"].tolist() == ["5"] and failed["impostor_pid"].iloc[0] == "3"
    participants = pd.read_csv(tmp_path / "evaluation" / "participants.csv")
    assert participants.query("group == 'query' and variable == 'female'")["value"].iloc[0] == 2
    assert table["n_impostor_pairs"].dtype.kind == "i"  # integer columns stay integers in the CSV
    with pytest.raises(ValueError, match="every gallery and query pid"):
        evaluate.participants_table(pd.read_csv(tmp_path / "matching" / "cohort.csv", dtype=str),
                                    pd.DataFrame({"pid": ["1"], "age": ["60"], "gender": ["1"], "race": ["1"]}))
    strata = pd.read_csv(tmp_path / "evaluation" / "strata.csv")
    assert strata[strata["stratum"].eq("sex_female")]["n"].tolist() == [2, 2, 2]
    for name, value in (("SIZES", (2, 4, 40)), ("N_QUERIES", 2), ("REPEATS", 3)):  # 40 exceeds the gallery
        monkeypatch.setattr(gallery_size, name, value)
    with pytest.raises(ValueError, match="number of queries"):
        cohort = pd.read_csv(tmp_path / "matching" / "cohort.csv", dtype=str)
        gallery_size.sensitivity(pd.DataFrame(), cohort, (2,), 6, 1)
    gallery_size.run(argparse.Namespace(matching_dir=str(tmp_path / "matching"),
                                        features=str(tmp_path / "features.csv"), out_dir=str(tmp_path / "size")))
    curve = pd.read_csv(tmp_path / "size" / "gallery_size.csv")
    assert sorted(curve["gallery_size"].unique()) == [2, 4, 6] and len(curve) == 3 * 3 * 3
    assert curve["rank1_rate"].between(0, 1).all() and (tmp_path / "size" / "gallery_size.tiff").exists()
    with open(tmp_path / "matching" / "cohort.csv", "a", encoding="utf-8") as stream:
        stream.write("9,T0,gallery,13,sternum,GE,Model,soft,2.5\n")
    with pytest.raises(ValueError, match="cohort.csv changed"):
        gallery_size.run(argparse.Namespace(matching_dir=str(tmp_path / "matching"),
                                            features=str(tmp_path / "features.csv"), out_dir=str(tmp_path / "size2")))
