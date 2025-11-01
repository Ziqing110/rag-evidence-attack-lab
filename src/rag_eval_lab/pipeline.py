import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd
from tqdm import tqdm

from .attack import (
    build_context_relevant_paragraphs,
    build_context_with_distractors,
    build_context_with_surrounding_paragraphs,
)
from .data import build_papers_index, filter_answerable_qas, load_peerqa_files, sample_qas
from .metrics import evaluate_prediction_rows
from .qa import safe_call_qa_model


def _save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_evidence_attack_pipeline(
    qa_path: str,
    papers_path: str,
    output_csv: str,
    output_jsonl: str,
    n: int = 20,
    seed: int = 451,
    qa_call_fn: Callable[[str, str], str] | None = None,
    client: Any | None = None,
    model: str | None = None,
    max_retries: int = 3,
    sleep_s: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """
    Run end-to-end evidence-removal evaluation pipeline.

    If qa_call_fn is provided, it is used directly as qa_call_fn(question, context) -> answer.
    Otherwise, OpenAI safe_call_qa_model is used with client+model.
    """
    qas, papers = load_peerqa_files(qa_path=qa_path, papers_path=papers_path)
    paper_index = build_papers_index(papers)

    available_paper_ids = list(paper_index["paper_id_to_idxs"].keys())
    filtered = filter_answerable_qas(qas, available_paper_ids=available_paper_ids)
    selected_qas = sample_qas(filtered, n=n, seed=seed)

    qa_call = _resolve_qa_call_fn(
        qa_call_fn=qa_call_fn,
        client=client,
        model=model,
        max_retries=max_retries,
        sleep_s=sleep_s,
    )

    results: List[Dict[str, Any]] = []
    for qa in tqdm(selected_qas, desc="Running evidence attack eval"):
        pack = build_context_relevant_paragraphs(qa, paper_index)
        qtext = qa.get("question", "")
        gold = qa.get("answer_free_form", "")

        pred_orig = qa_call(qtext, pack["context_original"])
        pred_pert = qa_call(qtext, pack["context_perturbed"])

        results.append(
            {
                "paper_id": qa.get("paper_id"),
                "question_id": qa.get("question_id"),
                "question": qtext,
                "answer_free_form": gold,
                "evidence_idxs": pack["evidence_idxs"],
                "original_idxs": pack["original_idxs"],
                "perturbed_idxs": pack["perturbed_idxs"],
                "context_original": pack["context_original"],
                "context_perturbed": pack["context_perturbed"],
                "pred_original": pred_orig,
                "pred_perturbed": pred_pert,
            }
        )

    # Save artifacts
    p_csv = Path(output_csv)
    p_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(p_csv, index=False)
    _save_jsonl(output_jsonl, results)

    original = evaluate_prediction_rows(results, pred_key="pred_original", gold_key="answer_free_form")
    perturbed = evaluate_prediction_rows(results, pred_key="pred_perturbed", gold_key="answer_free_form")
    return {"original": original, "perturbed": perturbed}


def _resolve_qa_call_fn(
    qa_call_fn: Callable[[str, str], str] | None,
    client: Any | None,
    model: str | None,
    max_retries: int,
    sleep_s: float,
) -> Callable[[str, str], str]:
    if qa_call_fn is not None:
        return qa_call_fn
    if client is None or model is None:
        raise ValueError("Either qa_call_fn or both client/model must be provided.")

    def _call(question: str, context: str) -> str:
        return safe_call_qa_model(
            question=question,
            context=context,
            client=client,
            model=model,
            max_retries=max_retries,
            sleep_s=sleep_s,
        )

    return _call


def run_context_robustness_pipeline(
    qa_path: str,
    papers_path: str,
    output_csv: str,
    output_jsonl: str,
    strategy: str,
    n: int = 20,
    seed: int = 451,
    qa_call_fn: Callable[[str, str], str] | None = None,
    client: Any | None = None,
    model: str | None = None,
    max_retries: int = 3,
    sleep_s: float = 1.0,
    radius: int = 1,
    total_sentences: int = 20,
    distractor_seed: int | None = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Run robustness evaluation with alternative context construction strategies.

    strategy:
    - "surrounding_paragraphs"
    - "distractor_mixed"
    """
    qas, papers = load_peerqa_files(qa_path=qa_path, papers_path=papers_path)
    paper_index = build_papers_index(papers)

    available_paper_ids = list(paper_index["paper_id_to_idxs"].keys())
    filtered = filter_answerable_qas(qas, available_paper_ids=available_paper_ids)
    selected_qas = sample_qas(filtered, n=n, seed=seed)

    qa_call = _resolve_qa_call_fn(
        qa_call_fn=qa_call_fn,
        client=client,
        model=model,
        max_retries=max_retries,
        sleep_s=sleep_s,
    )

    results: List[Dict[str, Any]] = []
    for qa in tqdm(selected_qas, desc=f"Running context robustness eval ({strategy})"):
        if strategy == "surrounding_paragraphs":
            pack = build_context_with_surrounding_paragraphs(qa, paper_index, radius=radius)
        elif strategy == "distractor_mixed":
            pack = build_context_with_distractors(
                qa,
                paper_index,
                total_sentences=total_sentences,
                seed=distractor_seed,
            )
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        qtext = qa.get("question", "")
        gold = qa.get("answer_free_form", "")
        pred_orig = qa_call(qtext, pack["context_original"])
        pred_pert = qa_call(qtext, pack["context_perturbed"])

        results.append(
            {
                "strategy": strategy,
                "paper_id": qa.get("paper_id"),
                "question_id": qa.get("question_id"),
                "question": qtext,
                "answer_free_form": gold,
                "evidence_idxs": pack.get("evidence_idxs", []),
                "original_idxs": pack.get("original_idxs", []),
                "perturbed_idxs": pack.get("perturbed_idxs", []),
                "context_original": pack.get("context_original", ""),
                "context_perturbed": pack.get("context_perturbed", ""),
                "pred_original": pred_orig,
                "pred_perturbed": pred_pert,
            }
        )

    p_csv = Path(output_csv)
    p_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(p_csv, index=False)
    _save_jsonl(output_jsonl, results)

    original = evaluate_prediction_rows(results, pred_key="pred_original", gold_key="answer_free_form")
    perturbed = evaluate_prediction_rows(results, pred_key="pred_perturbed", gold_key="answer_free_form")
    return {"original": original, "perturbed": perturbed}

