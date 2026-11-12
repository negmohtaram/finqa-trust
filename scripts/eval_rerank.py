"""Rerank the top 50 of each search method with a cross-encoder and compare on the same questions.

Usage: python scripts/eval_rerank.py
"""
import json
import time

from eval_retrieval import RESULTS, diff_cell, evidence_kind, load_chunks, mean, score_run

from finqa_trust.chunks import load_pages
from finqa_trust.retrieval import rerank

CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L6-v2"
SETS = [("table_whole", "fusion"), ("table_whole", "dense")]


def saved_rankings() -> dict[tuple[str, str, str], list[int]]:
    out = {}
    for line in (RESULTS / "retrieval_rankings.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["scope"] == "company":
            out[row["strategy"], row["method"], row["id"]] = row["ranking"]
    return out


def main() -> None:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(CROSS_ENCODER, device="cpu")
    model.model.double()  # this model returns NaN in 32-bit floats on this torch build
    score_pairs = lambda pairs: model.predict(pairs, batch_size=64, show_progress_bar=False)  # noqa: E731

    items = load_pages("dev")
    saved = saved_rankings()
    lines = [
        "# Reranking the top 50", "",
        f"{len(items)} FinQA dev questions, search inside the company, cross-encoder `{CROSS_ENCODER}`. "
        "Differences are paired on the same questions, in points, and * means the interval excludes 0.", "",
        "| chunking | candidates | ranking | recall@1 | recall@3 | recall@5 | recall@10 | complete@5 | MRR "
        "| recall@1 change | recall@5 change | MRR change | seconds per question |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    kinds = {}
    with (RESULTS / "rerank_rankings.jsonl").open("w") as out:
        for strategy, method in SETS:
            chunks = load_chunks(strategy)
            before = [saved[strategy, method, item["id"]] for item in items]
            start = time.perf_counter()
            after = [rerank(r, chunks, item["qa"]["question"], score_pairs) for r, item in zip(before, items)]
            seconds = (time.perf_counter() - start) / len(items)
            for item, ranking in zip(items, after):
                out.write(json.dumps({"id": item["id"], "strategy": strategy, "method": method + "_rerank",
                                      "ranking": ranking}) + "\n")
            old, new = score_run(before, chunks, items), score_run(after, chunks, items)
            kinds[strategy, method] = (old, new)
            for name, row in (("as searched", old), ("reranked", new)):
                changes = "|  |  |  |  |" if row is old else (
                    f"| {diff_cell(old['recall@1'], new['recall@1'])} | {diff_cell(old['recall@5'], new['recall@5'])} "
                    f"| {diff_cell(old['mrr'], new['mrr'])} | {seconds:.2f} |")
                lines.append(
                    f"| {strategy} | {method} | {name} | {mean(row['recall@1']):.3f} | {mean(row['recall@3']):.3f} "
                    f"| {mean(row['recall@5']):.3f} | {mean(row['recall@10']):.3f} | {mean(row['complete@5']):.3f} "
                    f"| {mean(row['mrr']):.3f} {changes}")

    lines += ["", "## Does the first search method still matter after reranking?", ""]
    fusion, dense = kinds["table_whole", "fusion"][1], kinds["table_whole", "dense"][1]
    for metric in ("recall@1", "recall@5", "mrr"):
        lines.append(f"- table_whole, reranked fusion minus reranked dense, {metric}: "
                     f"{diff_cell(dense[metric], fusion[metric])}")

    lines += ["", "## By where the evidence sits (table_whole, fusion, recall@5)", "",
              "| evidence | questions | as searched | reranked |", "|---|---|---|---|"]
    old, new = kinds["table_whole", "fusion"]
    for kind in ("table", "text", "both"):
        index = [i for i, item in enumerate(items) if evidence_kind(item) == kind]
        lines.append(f"| {kind} | {len(index)} | {mean([old['recall@5'][i] for i in index]):.3f} "
                     f"| {mean([new['recall@5'][i] for i in index]):.3f} |")
    (RESULTS / "rerank.md").write_text("\n".join(lines) + "\n")
    print("wrote results/rerank.md")


if __name__ == "__main__":
    main()
