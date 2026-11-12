"""Answer the same questions from retrieved chunks instead of the full page, and split the loss.

Full-page answers come from run_baseline.py. Rerunning replays finished calls for free.
Usage: python scripts/run_retrieved.py [--method fusion] [--runs 2] [--pause 9]
"""
import argparse
import json
from collections import Counter

from eval_retrieval import RESULTS, diff_cell, load_chunks, mean

from finqa_trust.chunks import load_pages
from finqa_trust.data import load_all
from finqa_trust.labels import label
from finqa_trust.model import DEFAULT_MODEL, DailyLimit, GroqClient, ModelRequest, ResumingClient, ask
from finqa_trust.prompts import retrieved_messages
from finqa_trust.retrieval import recall_at_k
from finqa_trust.sample import SAMPLES, load_ids


def rankings(method: str) -> dict[str, list[int]]:
    path = "rerank_rankings.jsonl" if method.endswith("_rerank") else "retrieval_rankings.jsonl"
    rows = [json.loads(line) for line in (RESULTS / path).read_text().splitlines()]
    return {r["id"]: r["ranking"] for r in rows if r["method"] == method and r["strategy"] == "table_whole"
            and r.get("scope", "company") == "company"}


def per_question(rows: list[dict]) -> dict[str, float]:
    """Share of a question's runs that were correct. Runs of one question are not independent."""
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        grouped.setdefault(row["id"], []).append(row["correct"])
    return {key: mean(flags) for key, flags in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="fusion")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--pause", type=float, default=9.0, help="seconds between live calls")
    args = parser.parse_args()

    records = [r for r in load_ids(SAMPLES / "dev400_arith100.json", load_all("dev")) if r.source == "finqa"]
    items = {item["id"]: item for item in load_pages("dev")}
    chunks, ranked = load_chunks("table_whole"), rankings(args.method)
    client = ResumingClient(GroqClient(), RESULTS / "recordings")

    rows = []
    try:
        for run in range(args.runs):
            for n, record in enumerate(records, start=1):
                key = record.id.removeprefix("finqa/")
                top = ranked[key][:5]
                request = ModelRequest(model=DEFAULT_MODEL, temperature=args.temperature, reasoning_effort="low",
                                       messages=retrieved_messages(record.question, [chunks[i].text for i in top]),
                                       json_mode=True, run=run)
                response = ask(client, request, args.pause)
                rows.append({"id": record.id, "run": run, "reply": response.text,
                             "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                             **label(response.text, record)})
                print(f"run {run} item {n:3}/{len(records)}  {rows[-1]['label']:10}  live {client.live_calls}", flush=True)
    except DailyLimit as err:
        print(f"Stopped at the daily limit after {client.live_calls} live calls, rerun later to resume: {err}")
        return

    (RESULTS / "retrieved_runs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    baseline = [json.loads(line) for line in (RESULTS / f"baseline_runs_t{args.temperature}.jsonl").read_text().splitlines()]
    page_rows = [r for r in baseline if r["source"] == "finqa" and r["run"] < args.runs]
    page, retrieved = per_question(page_rows), per_question(rows)

    def holds_all(record) -> bool:
        key = record.id.removeprefix("finqa/")
        item = items[key]
        return recall_at_k(ranked[key], chunks, list(item["qa"]["gold_inds"]), item["filename"], 5) == 1

    lines = [
        "# Full page against retrieved chunks", "",
        f"{len(records)} FinQA questions, {args.runs} runs each, same model and answer format. Retrieved means the "
        f"top 5 chunks of `{args.method}` on table_whole, inside the company. Differences are paired on the same "
        "questions, in points, and * means the interval excludes 0.", "",
        "| questions | n | full page | retrieved | retrieved minus full page |", "|---|---|---|---|---|",
    ]
    groups = [("all", records), ("all evidence in the top 5", [r for r in records if holds_all(r)]),
              ("some evidence missing", [r for r in records if not holds_all(r)])]
    for name, group in groups:
        a, b = [page[r.id] for r in group], [retrieved[r.id] for r in group]
        cell = diff_cell(a, b) if group else ""
        lines.append(f"| {name} | {len(group)} | {mean(a) if a else 0:.3f} | {mean(b) if b else 0:.3f} | {cell} |")

    pairs = Counter()
    by_key = {(r["id"], r["run"]): r for r in page_rows}
    for row in rows:
        before = by_key[row["id"], row["run"]]
        pairs["right" if before["correct"] else "wrong", "right" if row["correct"] else "wrong"] += 1
    lines += ["", "## Same question and run", "", "| full page | retrieved | answers |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {pairs[a, b]} |" for a in ("right", "wrong") for b in ("right", "wrong")]

    lines += ["", "## What broke, share of answers", "", "| label | full page | retrieved |", "|---|---|---|"]
    before, after = Counter(r["label"] for r in page_rows), Counter(r["label"] for r in rows)
    lines += [f"| {name} | {before[name] / len(page_rows):.3f} | {after[name] / len(rows):.3f} |"
              for name in sorted(set(before) | set(after))]
    (RESULTS / "gold_vs_retrieved.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n{client.live_calls} live calls, {sum(r['input_tokens'] + r['output_tokens'] for r in rows)} tokens")


if __name__ == "__main__":
    main()
