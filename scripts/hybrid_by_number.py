"""Does fusion gain more on questions that contain a number? Uses the saved rankings, no model calls.

Usage: python scripts/hybrid_by_number.py   (after eval_retrieval.py)
"""
import json
import re

from eval_retrieval import RESULTS, diff_cell, load_chunks, mean

from finqa_trust.chunks import load_pages
from finqa_trust.retrieval import recall_at_k


def main() -> None:
    items = load_pages("dev")
    saved = {}
    for line in (RESULTS / "retrieval_rankings.jsonl").read_text().splitlines():
        row = json.loads(line)
        saved[row["scope"], row["strategy"], row["method"], row["id"]] = row["ranking"]
    groups = {"has a number": [], "no number": []}
    for item in items:
        groups["has a number" if re.search(r"\d", item["qa"]["question"]) else "no number"].append(item)

    lines = ["# Fusion gain by question type", "",
             "Recall@5. Differences are paired on the same questions, in points, and * means the interval excludes 0.",
             "", "| search | chunking | question | n | dense | fusion | fusion minus dense |", "|---|---|---|---|---|---|---|"]
    for scope, name in (("company", "inside the company"), ("global", "all companies")):
        for strategy in ("fixed", "table_whole"):
            chunks = load_chunks(strategy)
            for kind, group in groups.items():
                dense, fusion = (
                    [recall_at_k(saved[scope, strategy, method, item["id"]], chunks, list(item["qa"]["gold_inds"]),
                                 item["filename"], 5) for item in group]
                    for method in ("dense", "fusion"))
                lines.append(f"| {name} | {strategy} | {kind} | {len(group)} | {mean(dense):.3f} | {mean(fusion):.3f} "
                             f"| {diff_cell(dense, fusion)} |")
    (RESULTS / "hybrid_by_number.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
