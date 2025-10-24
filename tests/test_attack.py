from rag_eval_lab.attack import (
    build_context_relevant_paragraphs,
    build_context_with_distractors,
    build_context_with_surrounding_paragraphs,
)


def _paper_index_fixture():
    # paper p1 sentence order is idx 0..5
    paper_id_to_idxs = {"p1": [0, 1, 2, 3, 4, 5]}
    paper_idx_to_text = {
        "p1": {
            0: "p0 s0",
            1: "p0 s1",
            2: "p1 s0",
            3: "p1 s1",
            4: "p2 s0",
            5: "p2 s1",
        }
    }
    # paragraph mapping: {0,1}->{p0}, {2,3}->{p1}, {4,5}->{p2}
    paper_idx_to_pidx = {"p1": {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}}
    return {
        "paper_id_to_idxs": paper_id_to_idxs,
        "paper_idx_to_text": paper_idx_to_text,
        "paper_idx_to_pidx": paper_idx_to_pidx,
    }


def test_relevant_paragraph_attack():
    paper_index = _paper_index_fixture()
    qa = {
        "paper_id": "p1",
        "answer_evidence_mapped": [{"idx": [2]}],  # evidence in paragraph 1
    }
    out = build_context_relevant_paragraphs(qa, paper_index)

    assert out["evidence_idxs"] == [2]
    assert out["original_idxs"] == [2, 3]
    assert out["perturbed_idxs"] == [3]
    assert "p1 s0" in out["context_original"]
    assert "p1 s0" not in out["context_perturbed"]


def test_surrounding_paragraphs_radius_1():
    paper_index = _paper_index_fixture()
    qa = {
        "paper_id": "p1",
        "answer_evidence_mapped": [{"idx": [2]}],  # paragraph 1
    }
    out = build_context_with_surrounding_paragraphs(qa, paper_index, radius=1)

    # selected paragraphs should cover all 0,1,2 for this small example
    assert out["selected_pidxs"] == [0, 1, 2]
    assert out["original_idxs"] == [0, 1, 2, 3, 4, 5]
    assert 2 not in out["perturbed_idxs"]


def test_surrounding_paragraphs_no_evidence():
    paper_index = _paper_index_fixture()
    qa = {"paper_id": "p1", "answer_evidence_mapped": []}
    out = build_context_with_surrounding_paragraphs(qa, paper_index, radius=1)
    assert out["original_idxs"] == []
    assert out["perturbed_idxs"] == []
    assert out["context_original"] == ""
    assert out["context_perturbed"] == ""


def test_distractor_builder_is_deterministic():
    paper_index = _paper_index_fixture()
    qa = {
        "paper_id": "p1",
        "answer_evidence_mapped": [{"idx": [2, 4]}],
    }
    out1 = build_context_with_distractors(qa, paper_index, total_sentences=4, seed=7)
    out2 = build_context_with_distractors(qa, paper_index, total_sentences=4, seed=7)
    assert out1["original_idxs"] == out2["original_idxs"]
    assert out1["perturbed_idxs"] == out2["perturbed_idxs"]
    # perturbed should not include evidence
    assert 2 not in out1["perturbed_idxs"]
    assert 4 not in out1["perturbed_idxs"]

