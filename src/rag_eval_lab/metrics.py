from collections.abc import Sequence


def compute_em(predictions: Sequence[str], references: Sequence[str]) -> float:
    """
    Placeholder for exact match metric.
    """
    raise NotImplementedError


def compute_f1(predictions: Sequence[str], references: Sequence[str]) -> float:
    """
    Placeholder for token-level F1 metric.
    """
    raise NotImplementedError

