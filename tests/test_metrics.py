import math

import pytest

from rag_eval_lab.metrics import (
    clean_pred_text,
    compute_em,
    compute_f1,
    evaluate_prediction_rows,
    exact_match,
    f1_score,
    normalize_text,
)


def test_normalize_and_exact_match():
    assert normalize_text("  Hello,  World! ") == "hello world"
    assert exact_match("Answer", "answer") == 1.0
    assert exact_match("A", "B") == 0.0


def test_f1_score():
    assert math.isclose(f1_score("a b c", "a b c d"), 0.8571428571428571, rel_tol=1e-6)
    assert f1_score("", "") == 1.0
    assert f1_score("", "x") == 0.0


def test_compute_em_f1_dataset():
    preds = ["x", "a b c", "no"]
    refs = ["x", "a b c d", "yes"]
    assert math.isclose(compute_em(preds, refs), 1 / 3, rel_tol=1e-6)
    assert compute_f1(preds, refs) > 0.3


def test_compute_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        compute_em(["a"], [])
    with pytest.raises(ValueError):
        compute_f1(["a"], [])


def test_clean_pred_text():
    assert clean_pred_text("Answer: 42") == "42"
    assert clean_pred_text("NOT_FOUND") == ""
    assert clean_pred_text(None) == ""


def test_evaluate_prediction_rows():
    rows = [
        {"pred_original": "Answer: Yes", "answer_free_form": "Yes"},
        {"pred_original": "NOT_FOUND", "answer_free_form": "No"},
    ]
    r = evaluate_prediction_rows(rows, pred_key="pred_original", gold_key="answer_free_form")
    assert r["N"] == 2.0
    assert math.isclose(r["EM"], 0.5, rel_tol=1e-6)
    assert r["NOT_FOUND_rate"] == 0.5

