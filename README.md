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

Project Status
--------------

This repository is being converted from a notebook workflow into a reusable Python package plus CLI scripts with tests.

CLI
---

Run evidence-removal evaluation:

`python scripts/run_evidence_attack_eval.py --qa-path PeerQA/data/qa.jsonl --papers-path PeerQA/data/papers.jsonl --n 20`

