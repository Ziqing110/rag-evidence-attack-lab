from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_eval_lab.reporting import build_comparison_table


def _parse_input_items(values: list[str]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for v in values:
        if "=" not in v:
            raise ValueError(f"Invalid --item format: '{v}'. Expected label=path.csv")
        label, path = v.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid --item format: '{v}'. Expected label=path.csv")
        items.append((label, path))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and compare multiple robustness result CSVs.")
    parser.add_argument(
        "--item",
        action="append",
        default=[],
        help="Input item in form label=path.csv (repeatable).",
    )
    parser.add_argument("--output-csv", default="", help="Optional output comparison CSV path.")
    parser.add_argument("--output-json", default="", help="Optional output comparison JSON path.")
    args = parser.parse_args()

    if not args.item:
        raise ValueError("Provide at least one --item label=path.csv")

    items = _parse_input_items(args.item)
    table = build_comparison_table(items)

    print(table.to_string(index=False))

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, index=False)

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(table.to_dict(orient="records"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

