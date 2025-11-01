import json
from pathlib import Path

import pandas as pd

from rag_eval_lab.pipeline import run_evidence_attack_pipeline


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_run_evidence_attack_pipeline_with_fake_qa(tmp_path):
    qa_path = tmp_path / "qa.jsonl"
    papers_path = tmp_path / "papers.jsonl"
    out_csv = tmp_path / "out.csv"
    out_jsonl = tmp_path / "out.jsonl"

    qas = [
        {
            "paper_id": "p1",
            "question_id": "q1",
            "question": "What is the keyword?",
            "answerable_mapped": True,
            "answer_evidence_mapped": [{"idx": [2]}],
            "answer_free_form": "keyword",
        }
    ]
    papers = [
        {"paper_id": "p1", "idx": 0, "pidx": 0, "sidx": 0, "content": "distractor"},
        {"paper_id": "p1", "idx": 1, "pidx": 0, "sidx": 1, "content": "more distractor"},
        {"paper_id": "p1", "idx": 2, "pidx": 1, "sidx": 0, "content": "keyword"},
        {"paper_id": "p1", "idx": 3, "pidx": 1, "sidx": 1, "content": "supporting text"},
    ]
    _write_jsonl(qa_path, qas)
    _write_jsonl(papers_path, papers)

    def fake_qa(question: str, context: str) -> str:
        return "keyword" if "keyword" in context else "NOT_FOUND"

    report = run_evidence_attack_pipeline(
        qa_path=str(qa_path),
        papers_path=str(papers_path),
        output_csv=str(out_csv),
        output_jsonl=str(out_jsonl),
        n=1,
        seed=451,
        qa_call_fn=fake_qa,
    )

    assert out_csv.exists()
    assert out_jsonl.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert report["original"]["EM"] == 1.0
    assert report["perturbed"]["EM"] == 0.0

