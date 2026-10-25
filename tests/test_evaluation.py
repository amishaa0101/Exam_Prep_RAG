import pytest

from evaluation import (
    TEST_CASES, UNANSWERABLE_CASES, load_gold, ndcg_at_k, hit_rate_at_k, is_relevant, keyword_recall, parse_judge_json, reciprocal_rank,
)


def test_hit_rate_at_k():
    flags = [False, False, True, False]
    assert hit_rate_at_k(flags, 1) == 0.0
    assert hit_rate_at_k(flags, 3) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank([True, False]) == 1.0
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([False, False]) == 0.0
    assert reciprocal_rank([]) == 0.0


def test_is_relevant_case_insensitive():
    assert is_relevant("A relation is in BCNF if...", ["bcnf"])
    assert not is_relevant("Merge sort is stable", ["bcnf", "3nf"])


def test_keyword_recall():
    recall, found, missing = keyword_recall(["Atomicity and Durability"], ["atomicity", "isolation"])
    assert recall == 0.5
    assert found == ["atomicity"] and missing == ["isolation"]


def test_parse_judge_json_tolerates_code_fences():
    text = '```json\n{"faithfulness": 5, "relevancy": 4, "completeness": 3}\n```'
    assert parse_judge_json(text)["relevancy"] == 4
    with pytest.raises(ValueError):
        parse_judge_json("no json here")


def test_test_cases_well_formed():
    ids = [c["id"] for c in TEST_CASES]
    assert len(ids) == len(set(ids))
    for case in TEST_CASES:
        for key in ("question", "relevant_keywords", "expected_keywords", "reference_points"):
            assert case[key], f"{case['id']} has empty {key}"


def test_ndcg_perfect_and_worst_order():
    grades = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], grades, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["c", "b", "a"], grades, 3) < ndcg_at_k(["a", "c", "b"], grades, 3)
    assert ndcg_at_k(["x", "y"], grades, 2) == 0.0  # unjudged counts as 0


def test_gold_file_covers_every_test_case():
    gold = load_gold()
    assert gold is not None
    assert set(gold) == {c["id"] for c in TEST_CASES}
    for qid, grades in gold.items():
        assert any(g == 2 for g in grades.values()), f"{qid} has no grade-2 chunk"
        assert set(grades.values()) <= {0, 1, 2}


def test_unanswerable_ids_distinct_from_answerable():
    assert not {c["id"] for c in UNANSWERABLE_CASES} & {c["id"] for c in TEST_CASES}
