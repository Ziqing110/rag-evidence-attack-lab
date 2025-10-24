import random
from typing import Any, Dict, List


def _extract_evidence_idxs(qa: Dict[str, Any]) -> List[int]:
    """
    Extract evidence sentence idxs from answer_evidence_mapped.
    Deduplicates while preserving first-seen order.
    """
    ev: List[int] = []
    seen = set()
    for m in qa.get("answer_evidence_mapped", []):
        for x in m.get("idx", []):
            try:
                i = int(x)
            except Exception:
                continue
            if i not in seen:
                ev.append(i)
                seen.add(i)
    return ev


def build_context_relevant_paragraphs(qa: Dict[str, Any], paper_index: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build original/perturbed contexts by selecting evidence-containing paragraphs.

    - original context: all sentences from paragraphs that contain evidence
    - perturbed context: original context with evidence sentences removed
    """
    paper_id = qa["paper_id"]
    paper_id_to_idxs = paper_index["paper_id_to_idxs"]
    paper_idx_to_text = paper_index["paper_idx_to_text"]
    paper_idx_to_pidx = paper_index["paper_idx_to_pidx"]

    evidence_idxs = _extract_evidence_idxs(qa)
    paper_idxs = paper_id_to_idxs[paper_id]

    evidence_pidxs = set()
    for i in evidence_idxs:
        p = paper_idx_to_pidx[paper_id].get(i)
        if p is not None:
            evidence_pidxs.add(p)

    original_idxs = [i for i in paper_idxs if paper_idx_to_pidx[paper_id].get(i) in evidence_pidxs]
    evidence_set = set(evidence_idxs)
    perturbed_idxs = [i for i in original_idxs if i not in evidence_set]

    context_original = "\n".join(paper_idx_to_text[paper_id][i] for i in original_idxs)
    context_perturbed = "\n".join(paper_idx_to_text[paper_id][i] for i in perturbed_idxs)

    return {
        "evidence_idxs": evidence_idxs,
        "original_idxs": original_idxs,
        "perturbed_idxs": perturbed_idxs,
        "context_original": context_original,
        "context_perturbed": context_perturbed,
    }


def build_context_with_surrounding_paragraphs(
    qa: Dict[str, Any],
    paper_index: Dict[str, Any],
    radius: int = 1,
) -> Dict[str, Any]:
    """
    Build contexts from evidence paragraphs plus surrounding paragraphs.

    - selected paragraphs = union of [p-radius, p+radius] for each evidence paragraph p
    - perturbed context removes only evidence sentences from selected context
    """
    paper_id = qa["paper_id"]
    paper_id_to_idxs = paper_index["paper_id_to_idxs"]
    paper_idx_to_text = paper_index["paper_idx_to_text"]
    paper_idx_to_pidx = paper_index["paper_idx_to_pidx"]

    evidence_idxs = _extract_evidence_idxs(qa)
    paper_idxs = paper_id_to_idxs[paper_id]

    evidence_pidxs = set()
    for i in evidence_idxs:
        p = paper_idx_to_pidx[paper_id].get(i)
        if p is not None:
            evidence_pidxs.add(p)

    # If no mapped evidence, return empty contexts.
    if not evidence_pidxs:
        return {
            "evidence_idxs": evidence_idxs,
            "original_idxs": [],
            "perturbed_idxs": [],
            "context_original": "",
            "context_perturbed": "",
            "radius": radius,
        }

    all_pidxs_in_paper = sorted(
        {
            p for i in paper_idxs
            for p in [paper_idx_to_pidx[paper_id].get(i)]
            if p is not None
        }
    )
    p_min = all_pidxs_in_paper[0]
    p_max = all_pidxs_in_paper[-1]

    selected_pidxs = set()
    for p in evidence_pidxs:
        lo = max(p_min, p - radius)
        hi = min(p_max, p + radius)
        for pp in range(lo, hi + 1):
            selected_pidxs.add(pp)

    original_idxs = [i for i in paper_idxs if paper_idx_to_pidx[paper_id].get(i) in selected_pidxs]
    evidence_set = set(evidence_idxs)
    perturbed_idxs = [i for i in original_idxs if i not in evidence_set]

    context_original = "\n".join(paper_idx_to_text[paper_id][i] for i in original_idxs)
    context_perturbed = "\n".join(paper_idx_to_text[paper_id][i] for i in perturbed_idxs)

    return {
        "evidence_idxs": evidence_idxs,
        "original_idxs": original_idxs,
        "perturbed_idxs": perturbed_idxs,
        "context_original": context_original,
        "context_perturbed": context_perturbed,
        "radius": radius,
        "evidence_pidxs": sorted(evidence_pidxs),
        "selected_pidxs": sorted(selected_pidxs),
    }


def build_context_with_distractors(
    qa: Dict[str, Any],
    paper_index: Dict[str, Any],
    total_sentences: int = 20,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Build contexts by mixing evidence with random distractor sentences.

    - original context: evidence + random non-evidence distractors
    - perturbed context: distractors only
    """
    paper_id = qa["paper_id"]
    paper_id_to_idxs = paper_index["paper_id_to_idxs"]
    paper_idx_to_text = paper_index["paper_idx_to_text"]

    evidence_idxs = _extract_evidence_idxs(qa)
    paper_idxs = paper_id_to_idxs[paper_id]
    evidence_set = set(evidence_idxs)

    pool = [i for i in paper_idxs if i not in evidence_set]
    k = max(0, total_sentences - len(evidence_idxs))
    rng = random.Random(seed) if seed is not None else random
    distractor_idxs = rng.sample(pool, k=min(k, len(pool)))

    original_idxs = sorted(set(evidence_idxs + distractor_idxs))
    perturbed_idxs = sorted(set(distractor_idxs))

    context_original = "\n".join(paper_idx_to_text[paper_id][i] for i in original_idxs)
    context_perturbed = "\n".join(paper_idx_to_text[paper_id][i] for i in perturbed_idxs)

    return {
        "evidence_idxs": evidence_idxs,
        "original_idxs": original_idxs,
        "perturbed_idxs": perturbed_idxs,
        "context_original": context_original,
        "context_perturbed": context_perturbed,
    }


def build_original_and_perturbed_context(qa: Dict[str, Any], paper_index: Dict[str, Any]) -> Dict[str, Any]:
    """
    Default context builder used for Part 1 behavior.
    """
    return build_context_relevant_paragraphs(qa, paper_index)

