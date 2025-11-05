"""
RAG Evidence Attack Lab package.
"""

from .attack import (
    build_context_relevant_paragraphs,
    build_context_with_distractors,
    build_context_with_surrounding_paragraphs,
)
from .data import build_papers_index, filter_answerable_qas, load_peerqa_files, sample_qas
from .metrics import compute_em, compute_f1, evaluate_prediction_rows, exact_match, f1_score
from .pipeline import run_context_robustness_pipeline, run_evidence_attack_pipeline
from .qa import build_openai_client_from_env, call_qa_model, safe_call_qa_model
from .reporting import build_comparison_table, summarize_results_csv

__all__ = [
    "load_peerqa_files",
    "build_papers_index",
    "filter_answerable_qas",
    "sample_qas",
    "build_context_relevant_paragraphs",
    "build_context_with_surrounding_paragraphs",
    "build_context_with_distractors",
    "call_qa_model",
    "safe_call_qa_model",
    "build_openai_client_from_env",
    "compute_em",
    "compute_f1",
    "exact_match",
    "f1_score",
    "evaluate_prediction_rows",
    "run_evidence_attack_pipeline",
    "run_context_robustness_pipeline",
    "summarize_results_csv",
    "build_comparison_table",
]

