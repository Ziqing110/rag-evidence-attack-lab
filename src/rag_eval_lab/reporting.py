from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from .metrics import evaluate_prediction_rows


def summarize_results_csv(path: str, label: str) -> pd.DataFrame:
    """
    Summarize one result CSV into two rows: original and perturbed.
    """
    df = pd.read_csv(path)
    rows = df.to_dict(orient="records")

    original = evaluate_prediction_rows(rows, pred_key="pred_original", gold_key="answer_free_form")
    perturbed = evaluate_prediction_rows(rows, pred_key="pred_perturbed", gold_key="answer_free_form")

    return pd.DataFrame(
        [
            {"setting": f"{label}_original", **original},
            {"setting": f"{label}_perturbed", **perturbed},
        ]
    )


def build_comparison_table(items: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Build a comparison table from [(label, csv_path), ...].
    """
    tables: List[pd.DataFrame] = []
    for label, path in items:
        tables.append(summarize_results_csv(path=path, label=label))
    if not tables:
        return pd.DataFrame(columns=["setting", "N", "EM", "F1", "NOT_FOUND_rate"])
    return pd.concat(tables, ignore_index=True)


def comparison_records(items: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """
    Return comparison table as JSON-serializable records.
    """
    return build_comparison_table(items).to_dict(orient="records")

