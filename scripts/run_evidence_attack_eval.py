from __future__ import annotations

import argparse
import json

from rag_eval_lab.pipeline import run_evidence_attack_pipeline
from rag_eval_lab.qa import build_openai_client_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evidence-removal robustness evaluation.")
    parser.add_argument("--qa-path", default="PeerQA/data/qa.jsonl", help="Path to PeerQA qa.jsonl")
    parser.add_argument("--papers-path", default="PeerQA/data/papers.jsonl", help="Path to PeerQA papers.jsonl")
    parser.add_argument("--n", type=int, default=20, help="Number of QA samples to evaluate")
    parser.add_argument("--seed", type=int, default=451, help="Random seed for sampling")
    parser.add_argument("--output-csv", default="out/evidence_attack_results.csv", help="Output CSV path")
    parser.add_argument("--output-jsonl", default="out/evidence_attack_results.jsonl", help="Output JSONL path")
    parser.add_argument("--output-report-json", default="", help="Optional report JSON path")
    args = parser.parse_args()

    client, model = build_openai_client_from_env()
    report = run_evidence_attack_pipeline(
        qa_path=args.qa_path,
        papers_path=args.papers_path,
        output_csv=args.output_csv,
        output_jsonl=args.output_jsonl,
        n=args.n,
        seed=args.seed,
        client=client,
        model=model,
    )

    print("Original:", json.dumps(report["original"], indent=2))
    print("Perturbed:", json.dumps(report["perturbed"], indent=2))

    if args.output_report_json:
        with open(args.output_report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

