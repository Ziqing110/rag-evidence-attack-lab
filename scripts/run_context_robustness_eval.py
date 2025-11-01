from __future__ import annotations

import argparse
import json

from rag_eval_lab.pipeline import run_context_robustness_pipeline
from rag_eval_lab.qa import build_openai_client_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run context-strategy robustness evaluation.")
    parser.add_argument("--qa-path", default="PeerQA/data/qa.jsonl", help="Path to PeerQA qa.jsonl")
    parser.add_argument("--papers-path", default="PeerQA/data/papers.jsonl", help="Path to PeerQA papers.jsonl")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["surrounding_paragraphs", "distractor_mixed"],
        help="Context construction strategy to evaluate.",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of QA samples to evaluate")
    parser.add_argument("--seed", type=int, default=451, help="Random seed for sampling questions")
    parser.add_argument("--radius", type=int, default=1, help="Radius for surrounding_paragraphs strategy")
    parser.add_argument(
        "--total-sentences",
        type=int,
        default=20,
        help="Total sentences for distractor_mixed strategy",
    )
    parser.add_argument(
        "--distractor-seed",
        type=int,
        default=0,
        help="Seed used for distractor sampling in distractor_mixed strategy",
    )
    parser.add_argument("--output-csv", default="out/context_robustness_results.csv", help="Output CSV path")
    parser.add_argument("--output-jsonl", default="out/context_robustness_results.jsonl", help="Output JSONL path")
    parser.add_argument("--output-report-json", default="", help="Optional report JSON path")
    args = parser.parse_args()

    client, model = build_openai_client_from_env()
    report = run_context_robustness_pipeline(
        qa_path=args.qa_path,
        papers_path=args.papers_path,
        output_csv=args.output_csv,
        output_jsonl=args.output_jsonl,
        strategy=args.strategy,
        n=args.n,
        seed=args.seed,
        client=client,
        model=model,
        radius=args.radius,
        total_sentences=args.total_sentences,
        distractor_seed=args.distractor_seed,
    )

    print("Original:", json.dumps(report["original"], indent=2))
    print("Perturbed:", json.dumps(report["perturbed"], indent=2))

    if args.output_report_json:
        with open(args.output_report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

