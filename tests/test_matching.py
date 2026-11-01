"""Roles from eligibility, gallery standardization, similarity of shared structures, ranking with ties and gaps."""
import math

import numpy as np
import pandas as pd
import pytest

from tests.conftest import load_stage_script

match = load_stage_script("04_matching/match_examinations.py")
STRUCTURES, FEATURES, THORACIC = match.STRUCTURES, match.FEATURES, match.THORACIC


def exam_features(pid, timepoint, values, structures=STRUCTURES):
    return [{"pid": pid, "timepoint": timepoint, "structure": name, **dict(zip(FEATURES, values[name], strict=True))}
            for name in structures]


@pytest.fixture
def cohort_and_features():
    """Gallery 1, 2, 3, 4, 6 (6 duplicates 1); queries 1 (copy of its T0), 2 (renumbered vertebrae), 3 (sternum)."""
    rng = np.random.default_rng(0)
    base = {pid: {name: rng.normal(size=len(FEATURES)) for name in STRUCTURES} for pid in "1234"}
    base["6"] = base["1"]
    shifted = {"sternum": base["2"]["sternum"],
               **{f"vertebrae_T{k}": base["2"][f"vertebrae_T{k + 1}"] for k in range(1, 12)}}
    rows = [row for pid in "12346" for row in exam_features(pid, "T0", base[pid])]
    rows += exam_features("1", "T2", base["1"]) + exam_features("2", "T2", shifted, ["sternum", *THORACIC[:11]])
    rows += exam_features("3", "T2", base["3"], ["sternum"])
    features = pd.DataFrame(rows)
    cohort = pd.DataFrame([{"pid": pid, "timepoint": tp, "role": role} for pid, tp, role in
                           [("1", "T0", "gallery"), ("2", "T0", "gallery"), ("3", "T0", "gallery"),
                            ("4", "T0", "gallery"), ("6", "T0", "gallery"),
                            ("1", "T2", "query"), ("2", "T2", "query"), ("3", "T2", "query")]])
    return cohort, features


def test_roles_follow_eligibility_and_selected_series_supply_acquisition_fields():
    def status(pid, name):  # pid 4 has only the sternum usable at T0 (numbering gap): excluded from every role
        return "usable" if name == "sternum" or (name != "vertebrae_T1" and pid != "4") else "absent"

    exams = [("1", "T0", "True"), ("1", "T2", "True"), ("2", "T2", "True"), ("3", "T0", "False"), ("4", "T0", "True"),
             ("4", "T2", "True")]
    qc = pd.DataFrame([{"pid": pid, "timepoint": tp, "structure": name, "exam_eligible": eligible,
                        "structure_status": status(pid, name) if eligible == "True" else "ineligible_exam"}
                       for pid, tp, eligible in exams for name in STRUCTURES])
    selection = pd.DataFrame([{"pid": pid, "timepoint": tp, "status": "selected", "manufacturer_group": "GE",
                               "ManufacturerModelName": "M", "kernel_class": "soft", "thickness_mm": "2.5"}
                              for pid, tp, _ in exams])
    cohort = match.assign_roles(qc, selection)
    assert cohort.set_index(["pid", "timepoint"])["role"].to_dict() == {
        ("1", "T0"): "gallery", ("1", "T2"): "query", ("2", "T2"): "no_reference",
        ("4", "T0"): "single_structure", ("4", "T2"): "single_structure"}
    assert cohort["n_usable"].tolist() == [12, 12, 12, 1, 1]
    assert cohort["usable_structures"][0].split(";") == [s for s in STRUCTURES if s != "vertebrae_T1"]
    with pytest.raises(ValueError, match="selected series"):
        match.assign_roles(qc, selection.iloc[1:])


def test_similarity_is_one_over_one_plus_distance_and_nan_without_the_structure():
    query = np.array([[0.0, 0.0], [np.nan, np.nan], [3.0, 4.0]])
    gallery = np.array([[0.0, 0.0], [np.nan, np.nan]])
    similarity = match.structure_similarity(query, gallery)
    assert similarity[0, 0] == 1 and similarity[2, 0] == pytest.approx(1 / 6)
    assert np.isnan(similarity[1]).all() and np.isnan(similarity[:, 1]).all()
    assert np.isnan(match.pair_similarity(query, gallery, np.array([0, 1]), np.array([0, 0]))).tolist() == [False, True]


def test_matching_ranks_with_unfavourable_ties_and_counts_unscorable_pairs_as_failures(cohort_and_features):
    cohort, features = cohort_and_features
    tables = match.match(features, cohort, chunk=2)
    results = tables["matching_results"].set_index(["pid", "method"])
    for method in match.METHODS:  # query 1 equals its T0, but gallery 6 duplicates it: rank 2, not 1
        own = results.loc[("1", method)]
        assert own["true_score"] == pytest.approx(1) and own["true_rank"] == 2
        assert own["top1_score"] == pytest.approx(1) and own["n_scored"] == 5
    assert results.loc[("1", "integrated"), "n_common"] == 13
    assert results.loc[("2", "sternum"), "true_rank"] == 1 and results.loc[("2", "vertebrae"), "true_rank"] > 1
    vertebrae_only = results.loc[("3", "vertebrae")]
    assert np.isnan(vertebrae_only["true_rank"]) and vertebrae_only["n_common"] == 0 and vertebrae_only["n_scored"] == 0
    assert vertebrae_only["top1_pid"] == "" and np.isnan(vertebrae_only["top1_score"])
    assert results.loc[("3", "integrated"), "n_common"] == 1 and results.loc[("3", "integrated"), "true_rank"] == 1
    structures = tables["query_structures"].set_index("pid")
    assert structures.loc["2", "vertebrae_shift_up"] == pytest.approx(1)  # query T_k equals the mate's T_(k+1)
    assert structures.loc["2", "vertebrae_shift_down"] < 1 and np.isnan(structures.loc["3", "vertebrae_shift_up"])
    assert structures.loc["1", [f"mate_{s}" for s in STRUCTURES]].to_numpy().tolist() == pytest.approx([1.0] * 13)
    assert structures.loc["3", "mate_vertebrae_T5"] != structures.loc["3", "mate_vertebrae_T5"]  # NaN: not shared
    assert structures.loc["1", "impostor_pid"] == "6"
    impostors = tables["impostor_histogram"].groupby("method")["count"].sum()
    scored = results["n_scored"] - results["true_score"].notna()
    assert impostors.to_dict() == scored.groupby(level="method").sum().to_dict()
    assert tables["n_gallery"] == 5 and tables["n_query"] == 3 and tables["seconds"] > 0


def test_scores_equal_the_hand_computed_definition():
    """Two structures, three gallery and two query examinations; every feature of a structure carries one value."""
    def values(sternum, t1=None):
        return {"sternum": np.full(14, sternum), **({} if t1 is None else {"vertebrae_T1": np.full(14, t1)})}

    both = ["sternum", "vertebrae_T1"]
    rows = exam_features("1", "T0", values(0), ["sternum"]) + exam_features("2", "T0", values(2, 10), both)
    rows += exam_features("3", "T0", values(4, 20), both) + exam_features("1", "T2", values(0.5, 10), both)
    rows += exam_features("2", "T2", values(4), ["sternum"])
    cohort = pd.DataFrame({"pid": ["1", "2", "3", "1", "2"], "timepoint": ["T0"] * 3 + ["T2"] * 2,
                           "role": ["gallery"] * 3 + ["query"] * 2})
    tables = match.match(pd.DataFrame(rows), cohort)
    sim = lambda dz: 1 / (1 + math.sqrt(14) * dz)  # noqa: E731  14 identical standardized differences
    results = tables["matching_results"].set_index(["pid", "method"])
    q1 = results.loc["1"]  # sternum z = -0.75 vs gallery z = -1, 0, 1; T1 equals gallery 2, gallery 1 has no T1
    assert q1.loc["sternum", "true_score"] == pytest.approx(sim(0.25)) and q1.loc["sternum", "true_rank"] == 1
    assert q1.loc["sternum", "top1_pid"] == "1" and q1.loc["sternum", "top2_score"] == pytest.approx(sim(0.75))
    assert np.isnan(q1.loc["vertebrae", "true_rank"]) and q1.loc["vertebrae", "n_scored"] == 2  # only own T0 unscorable
    assert q1.loc["vertebrae", "top1_pid"] == "2" and q1.loc["vertebrae", "top1_score"] == pytest.approx(1)
    assert q1.loc["integrated", "true_score"] == pytest.approx(sim(0.25)) and q1.loc["integrated", "n_common"] == 1
    assert q1.loc["integrated", "true_rank"] == 2
    assert q1.loc["integrated", "top1_score"] == pytest.approx((sim(0.75) + 1) / 2)  # gallery 2: two structures
    q2 = results.loc["2"]  # sternum only, z = 1: identical to gallery 3, one SD from its own T0
    assert q2.loc["sternum", "true_score"] == pytest.approx(sim(1)) and q2.loc["sternum", "true_rank"] == 2
    assert q2.loc["sternum", "top1_pid"] == "3" and q2.loc["sternum", "top1_score"] == 1
    assert np.isnan(q2.loc["vertebrae", "true_rank"]) and q2.loc["vertebrae", "n_scored"] == 0
    assert q2.loc["integrated", "true_rank"] == 2 and q2.loc["integrated", "top2_score"] == pytest.approx(sim(1))
    assert tables["impostor_histogram"].groupby("method")["count"].sum().to_dict() == {"integrated": 4, "sternum": 4,
                                                                                      "vertebrae": 2}
    structures = tables["query_structures"].set_index("pid")
    assert structures.loc["1", "mate_sternum"] == pytest.approx(sim(0.25))
    assert np.isnan(structures.loc["1", "mate_vertebrae_T1"]) and structures.loc["1", "impostor_pid"] == "2"
    assert structures.loc["1", "impostor_vertebrae_T1"] == pytest.approx(1)
    assert results.index.get_level_values("method").tolist()[:3] == ["sternum"] * 2 + ["vertebrae"]  # fixed order


def test_features_are_standardized_with_gallery_statistics(cohort_and_features):
    cohort, features = cohort_and_features
    scaling = match.gallery_scaling(match.feature_arrays(features, np.array(["1", "2", "3", "4", "6"]), "T0"))
    gallery = features[features["timepoint"].eq("T0") & features["structure"].eq("vertebrae_T3")]
    assert scaling["vertebrae_T3"][0][0] == pytest.approx(gallery["MeshVolume"].mean())
    assert scaling["vertebrae_T3"][1][0] == pytest.approx(gallery["MeshVolume"].std(ddof=1))
    constant = features.copy()
    constant.loc[constant["structure"].eq("sternum"), "Flatness"] = 0.5
    with pytest.raises(ValueError, match="Constant"):
        match.match(constant, cohort)
