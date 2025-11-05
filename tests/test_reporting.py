import json
from pathlib import Path

import pandas as pd

from rag_eval_lab.reporting import build_comparison_table, summarize_results_csv


def _write_csv(path: Path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_summarize_results_csv(tmp_path):
    p = tmp_path / "part1.csv"
    _write_csv(
        p,
        [
            {"pred_original": "A", "pred_perturbed": "NOT_FOUND", "answer_free_form": "A"},
            {"pred_original": "B", "pred_perturbed": "C", "answer_free_form": "B"},
        ],
    )
    t = summarize_results_csv(str(p), label="p1")
    assert len(t) == 2
    assert set(t["setting"]) == {"p1_original", "p1_perturbed"}


def test_build_comparison_table(tmp_path):
    p1 = tmp_path / "p1.csv"
    p2 = tmp_path / "p2.csv"
    _write_csv(
        p1,
        [{"pred_original": "x", "pred_perturbed": "NOT_FOUND", "answer_free_form": "x"}],
    )
    _write_csv(
        p2,
        [{"pred_original": "y", "pred_perturbed": "z", "answer_free_form": "y"}],
    )
    t = build_comparison_table([("p1", str(p1)), ("p2", str(p2))])
    assert len(t) == 4
    assert "NOT_FOUND_rate" in t.columns
    assert t["N"].sum() == 4.0

