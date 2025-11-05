RAG Evidence Attack Lab
=======================

Scientific QA Robustness Evaluation Pipeline
--------------------------------------------

This project evaluates how robust scientific QA systems are when supporting evidence is missing or perturbed.
Using PeerQA-style data, it compares model behavior under:

- original evidence-supported context
- evidence-removed (perturbed) context
- alternative context construction strategies for robustness stress testing

The pipeline reports Exact Match (EM), token-level F1, and abstention-style behavior (`NOT_FOUND`) to diagnose grounding failures.


Even if general RAG evaluation tools exist, this project provides a task-specific, reproducible robustness harness for scientific QA:

- controlled evidence-removal attacks tied to mapped evidence
- context construction variants to analyze failure modes
- result-aware scoring with abstention behavior tracking (`NOT_FOUND`)
- reproducible CSV/JSONL outputs for reporting and comparison

CLI
---

Run evidence-removal evaluation:

`python scripts/run_evidence_attack_eval.py --qa-path PeerQA/data/qa.jsonl --papers-path PeerQA/data/papers.jsonl --n 20`

Run context-strategy robustness evaluation:

`python scripts/run_context_robustness_eval.py --strategy surrounding_paragraphs --qa-path PeerQA/data/qa.jsonl --papers-path PeerQA/data/papers.jsonl --n 20`

`python scripts/run_context_robustness_eval.py --strategy distractor_mixed --total-sentences 20 --qa-path PeerQA/data/qa.jsonl --papers-path PeerQA/data/papers.jsonl --n 20`

Generate a consolidated comparison report:

`python scripts/run_results_report.py --item p1=out/evidence_attack_results.csv --item p2_para=out/context_para_results.csv --item p2_mix=out/context_mix_results.csv`

Academic Attribution
--------------------

This project implementation and experimental framing are inspired by concepts and assignments from **CISC451 at Queen's University** (multimodal/document QA course content).  
The codebase and engineering structure in this repository are independently implemented as a standalone portfolio project.

