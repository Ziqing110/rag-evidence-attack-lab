import re
import string
from collections import Counter
from collections.abc import Sequence
from typing import Any, Dict, List

import pandas as pd


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(predictions: Sequence[str], references: Sequence[str]) -> float:
    preds = list(predictions)
    refs = list(references)
    if len(preds) != len(refs):
        raise ValueError("predictions and references must have same length")
    if not preds:
        return 0.0
    return float(sum(exact_match(p, g) for p, g in zip(preds, refs)) / len(preds))


def compute_f1(predictions: Sequence[str], references: Sequence[str]) -> float:
    preds = list(predictions)
    refs = list(references)
    if len(preds) != len(refs):
        raise ValueError("predictions and references must have same length")
    if not preds:
        return 0.0
    return float(sum(f1_score(p, g) for p, g in zip(preds, refs)) / len(preds))


def clean_pred_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if s.lower().startswith("answer:"):
        s = s.split(":", 1)[1].strip()
    if s in {"NOT_FOUND", "not_found", "not found"}:
        return ""
    return s


def evaluate_prediction_rows(
    rows: List[Dict[str, Any]],
    pred_key: str,
    gold_key: str = "answer_free_form",
) -> Dict[str, float]:
    preds = [clean_pred_text(r.get(pred_key, "")) for r in rows]
    golds = [str(r.get(gold_key, "")) for r in rows]
    n = len(rows)
    if n == 0:
        return {"N": 0.0, "EM": 0.0, "F1": 0.0, "NOT_FOUND_rate": 0.0}
    return {
        "N": float(n),
        "EM": compute_em(preds, golds),
        "F1": compute_f1(preds, golds),
        "NOT_FOUND_rate": float(sum(1 for p in preds if not p.strip()) / n),
    }


def evaluate_from_csv(path: str) -> pd.DataFrame:
    """
    Evaluate a result CSV containing pred_original/pred_perturbed columns.
    """
    df = pd.read_csv(path)
    rows = df.to_dict(orient="records")
    original = evaluate_prediction_rows(rows, pred_key="pred_original", gold_key="answer_free_form")
    perturbed = evaluate_prediction_rows(rows, pred_key="pred_perturbed", gold_key="answer_free_form")
    return pd.DataFrame(
        [
            {"setting": "original", **original},
            {"setting": "perturbed", **perturbed},
        ]
    )

