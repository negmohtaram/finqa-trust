"""Give each retrieval miss one failure type, first matching rule wins, so the counts add up to all misses.

Usage: python scripts/retrieval_taxonomy.py   (after eval_retrieval.py)
"""
import json
import re
from collections import Counter

from eval_retrieval import RESULTS, load_chunks

from finqa_trust.chunks import holds_evidence, load_pages

TYPES = {
    "wrong page": "no chunk from the question's page in the top 5",
    "never found": "a missing piece is not in the top 50",
    "harmless sentence": "only sentences without figures are missing",
    "table without its sentence": "the table is found, a sentence that explains it is not",
    "sentence ranked low": "only sentences are missing, and no table evidence was found",
    "table ranked low": "the table is on ranks 6 to 50",
}


def failure_type(miss: dict, ranking: list[int], chunks: list, page: str) -> str:
    if not any(chunks[i].page == page for i in ranking[:5]):
        return "wrong page"
    for gold in miss["missing"]:
        if not any(chunks[i].page == page and holds_evidence(chunks[i], gold) for i in ranking):
            return "never found"
    if any(gold.startswith("table") for gold in miss["missing"]):
        return "table ranked low"
    if not any(re.search(r"\d", text) for text in miss["missing"].values()):
        return "harmless sentence"
    if any(gold.startswith("table") for gold in miss["found"]):
        return "table without its sentence"
    return "sentence ranked low"


def main() -> None:
    items = {item["id"]: item for item in load_pages("dev")}
    chunks = load_chunks("table_whole")
    rankings = {}
    for line in (RESULTS / "retrieval_rankings.jsonl").read_text().splitlines():
        row = json.loads(line)
        if (row["scope"], row["strategy"], row["method"]) == ("company", "table_whole", "fusion"):
            rankings[row["id"]] = row["ranking"]
    misses = [json.loads(line) for line in (RESULTS / "retrieval_misses.jsonl").read_text().splitlines()]
    labels = {m["id"]: failure_type(m, rankings[m["id"]], chunks, items[m["id"]]["filename"]) for m in misses}
    (RESULTS / "retrieval_taxonomy.jsonl").write_text(
        "".join(json.dumps({"id": key, "type": value}) + "\n" for key, value in labels.items()))

    counts = Counter(labels.values())
    print("| type | rule | count | share |\n|---|---|---|---|")
    for name, rule in TYPES.items():
        print(f"| {name} | {rule} | {counts[name]} | {counts[name] / len(misses):.0%} |")
    print(f"| total | | {sum(counts.values())} | |")


if __name__ == "__main__":
    main()
