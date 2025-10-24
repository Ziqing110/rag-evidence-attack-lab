import json
from pathlib import Path

from rag_eval_lab.data import (
    build_papers_index,
    filter_answerable_qas,
    load_jsonl,
    load_peerqa_files,
    sample_qas,
)


def _write_jsonl(path, rows):
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_jsonl_and_peerqa_files(tmp_path):
    qa_rows = [{"id": 1}, {"id": 2}]
    paper_rows = [{"paper_id": "p1", "idx": 0, "pidx": 0, "sidx": 0, "content": "text"}]
    qa_path = tmp_path / "qa.jsonl"
    papers_path = tmp_path / "papers.jsonl"
    _write_jsonl(qa_path, qa_rows)
    _write_jsonl(papers_path, paper_rows)

    assert load_jsonl(str(qa_path)) == qa_rows
    qas, papers = load_peerqa_files(str(qa_path), str(papers_path))
    assert qas == qa_rows
    assert papers == paper_rows


def test_build_papers_index_basic():
    rows = [
        {"paper_id": "p1", "idx": 1, "pidx": 0, "sidx": 1, "content": "B"},
        {"paper_id": "p1", "idx": 0, "pidx": 0, "sidx": 0, "content": "A"},
        {"paper_id": "p1", "idx": 2, "pidx": 1, "sidx": 0, "content": "C"},
        {"paper_id": "p2", "idx": 0, "pidx": 0, "sidx": 0, "content": "X"},
        {"paper_id": "", "idx": 3, "pidx": 0, "sidx": 0, "content": "bad"},
        {"paper_id": "p1", "idx": 9, "pidx": 9, "sidx": 9, "content": ""},
    ]

    idx = build_papers_index(rows)
    assert idx["stats"]["num_papers"] == 2
    assert idx["stats"]["missing_rows"] == 1
    assert idx["stats"]["empty_text_skipped"] == 1

    # stable order by pidx, sidx, idx
    assert idx["paper_id_to_idxs"]["p1"] == [0, 1, 2]
    assert idx["paper_idx_to_text"]["p1"][0] == "A"
    assert idx["paper_pidx_to_sent_idxs"]["p1"][0] == [0, 1]
    assert idx["paper_idx_to_pidx"]["p1"][2] == 1


def test_filter_and_sample_qas():
    qas = [
        {"paper_id": "p1", "answerable_mapped": True, "answer_evidence_mapped": [{"idx": [1]}], "answer_free_form": "x"},
        {"paper_id": "p2", "answerable_mapped": False, "answer_evidence_mapped": [{"idx": [1]}], "answer_free_form": "x"},
        {"paper_id": "p1", "answerable_mapped": True, "answer_evidence_mapped": [], "answer_free_form": "x"},
        {"paper_id": "p1", "answerable_mapped": True, "answer_evidence_mapped": [{"idx": [2]}], "answer_free_form": None},
        {"paper_id": "p3", "answerable_mapped": True, "answer_evidence_mapped": [{"idx": [3]}], "answer_free_form": "x"},
    ]

    filtered = filter_answerable_qas(qas, available_paper_ids=["p1", "p2"])
    assert len(filtered) == 1
    assert filtered[0]["paper_id"] == "p1"

    sampled = sample_qas(filtered * 4, n=2, seed=123)
    assert len(sampled) == 2
    sampled2 = sample_qas(filtered * 4, n=2, seed=123)
    assert sampled == sampled2

