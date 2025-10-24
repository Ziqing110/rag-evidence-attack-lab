import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dict records.
    """
    out: List[Dict[str, Any]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_peerqa_files(qa_path: str, papers_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load PeerQA qa/papers JSONL files.
    """
    return load_jsonl(qa_path), load_jsonl(papers_path)


def build_papers_index(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build paper-scoped sentence and paragraph indexes from papers.jsonl-like rows.

    Returns dict containing:
    - paper_to_sents: paper_id -> list[rec] sorted by (pidx, sidx, idx)
    - paper_idx_to_text: paper_id -> {idx -> text}
    - paper_id_to_idxs: paper_id -> [idx in stable order]
    - paper_pidx_to_sent_idxs: paper_id -> {pidx -> [idx,...]}
    - paper_idx_to_pidx: paper_id -> {idx -> pidx}
    - stats: counters for skipped/missing rows
    """
    paper_to_sents = defaultdict(list)
    paper_idx_to_text = defaultdict(dict)
    paper_pidx_to_sent_idxs = defaultdict(lambda: defaultdict(list))
    paper_idx_to_pidx = defaultdict(dict)

    missing = 0
    skipped_empty_text = 0

    for r in papers:
        pid = str(r.get("paper_id", "")).strip()
        if not pid:
            missing += 1
            continue

        idx = r.get("idx")
        pidx = r.get("pidx")
        sidx = r.get("sidx")
        if idx is None or pidx is None or sidx is None:
            missing += 1
            continue

        try:
            idx = int(idx)
            pidx = int(pidx)
            sidx = int(sidx)
        except Exception:
            missing += 1
            continue

        text = r.get("content", None)
        if text is None:
            text = r.get("text", None)
        if text is None:
            missing += 1
            continue

        text = str(text).strip()
        if not text:
            skipped_empty_text += 1
            continue

        rec = {
            "idx": idx,
            "pidx": pidx,
            "sidx": sidx,
            "text": text,
            "type": r.get("type", None),
            "last_heading": r.get("last_heading", None),
        }

        paper_to_sents[pid].append(rec)
        paper_idx_to_text[pid][idx] = text
        paper_pidx_to_sent_idxs[pid][pidx].append(idx)
        paper_idx_to_pidx[pid][idx] = pidx

    # finalize with stable ordering
    paper_id_to_idxs: Dict[str, List[int]] = {}
    for pid, lst in paper_to_sents.items():
        lst.sort(key=lambda x: (x["pidx"], x["sidx"], x["idx"]))
        paper_id_to_idxs[pid] = [x["idx"] for x in lst]

        pmap = defaultdict(list)
        for rec in lst:
            pmap[rec["pidx"]].append(rec["idx"])
        paper_pidx_to_sent_idxs[pid] = dict(pmap)

    return {
        "paper_to_sents": dict(paper_to_sents),
        "paper_idx_to_text": {k: dict(v) for k, v in paper_idx_to_text.items()},
        "paper_id_to_idxs": paper_id_to_idxs,
        "paper_pidx_to_sent_idxs": {k: dict(v) for k, v in paper_pidx_to_sent_idxs.items()},
        "paper_idx_to_pidx": {k: dict(v) for k, v in paper_idx_to_pidx.items()},
        "stats": {
            "num_papers": len(paper_id_to_idxs),
            "missing_rows": missing,
            "empty_text_skipped": skipped_empty_text,
        },
    }


def filter_answerable_qas(
    qas: List[Dict[str, Any]],
    available_paper_ids: List[str],
) -> List[Dict[str, Any]]:
    """
    Keep only QA examples that are answerable, mapped to evidence, have gold answers,
    and belong to available papers.
    """
    pid_set = set(available_paper_ids)
    return [
        q for q in qas
        if q.get("answerable_mapped") is True
        and q.get("answer_evidence_mapped")
        and q.get("paper_id") in pid_set
        and q.get("answer_free_form") is not None
    ]


def sample_qas(qas: List[Dict[str, Any]], n: int, seed: int = 451) -> List[Dict[str, Any]]:
    """
    Deterministically sample up to n QA records.
    """
    rng = random.Random(seed)
    if n >= len(qas):
        return list(qas)
    return rng.sample(qas, n)

