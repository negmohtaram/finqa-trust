"""Compare the approaches on the same questions: accuracy, paired difference against the model
alone, error labels, and what each costs per question and per correct answer.

Reads results/<approach>.jsonl written by run_approaches.py and writes results/comparison.md.
Usage: python scripts/compare.py
"""
import json
from pathlib import Path

from finqa_trust.labels import LABELS
from finqa_trust.stats import bootstrap_ci, paired_bootstrap_diff

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FILES = {"alone": "alone.jsonl", "code": "code.jsonl", "temp0": "temp0.jsonl", "vote k=3": "vote_k3.jsonl",
         "vote k=5": "vote_k5.jsonl"}
# Groq list price for openai/gpt-oss-120b (console.groq.com, checked 2026-10-17). The runs used the
# free tier, so these are what the same calls would cost on a paid account.
PRICE_IN, PRICE_OUT = 0.15 / 1e6, 0.60 / 1e6


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (RESULTS / name).read_text().splitlines()]


def pct(x: float) -> str:
    return f"{x:.0%}"


def main() -> None:
    runs = {a: load(f) for a, f in FILES.items() if (RESULTS / f).exists()}
    ids = [row["id"] for row in runs["alone"]]
    for approach, rows in runs.items():
        if [row["id"] for row in rows] != ids:
            raise SystemExit(f"{approach} was run on different questions; the comparison must be paired")
    base = [row["correct"] for row in runs["alone"]]
    n = len(ids)

    lines = [f"Paired comparison on the same {n} questions (model alone = temperature 1.0, one answer). "
             "Runs used the free tier; the price column is Groq's list price for the same tokens.", "",
             "| approach | correct | 95% CI | vs alone | 95% CI of difference | alone wrong → right | "
             "alone right → wrong | calls | tokens per question | tokens per correct answer | "
             "latency serial / parallel (s) | list price per 1,000 correct |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for approach, rows in runs.items():
        flags = [row["correct"] for row in rows]
        acc, lo, hi = bootstrap_ci(flags)
        diff, dlo, dhi = paired_bootstrap_diff(base, flags)
        fixed = sum(not a and b for a, b in zip(base, flags))
        broke = sum(a and not b for a, b in zip(base, flags))
        cost = sum(r["input_tokens"] * PRICE_IN + r["output_tokens"] * PRICE_OUT for r in rows)
        tokens = sum(r["input_tokens"] + r["output_tokens"] for r in rows) / n
        serial = sum(r["latency_serial_s"] for r in rows) / n
        parallel = sum(r["latency_parallel_s"] for r in rows) / n
        calls = sum(r["calls"] for r in rows) / n
        vs = "" if approach == "alone" else f"{diff:+.0%}"
        dci = "" if approach == "alone" else f"[{dlo:+.0%}, {dhi:+.0%}]"
        lines.append(f"| {approach} | {pct(acc)} | [{pct(lo)}, {pct(hi)}] | {vs} | {dci} | {fixed} | {broke} | "
                     f"{calls:.0f} | {tokens:,.0f} | {tokens * n / max(sum(flags), 1):,.0f} | {serial:.1f} / {parallel:.1f} | "
                     f"${cost / max(sum(flags), 1) * 1000:.2f} |")

    used = [label for label in LABELS if any(r["label"] == label for rows in runs.values() for r in rows)]
    lines += ["", "Answers by the first step that broke:", "",
              "| approach | " + " | ".join(used) + " |", "|---|" + "---|" * len(used)]
    for approach, rows in runs.items():
        lines.append(f"| {approach} | " + " | ".join(str(sum(r["label"] == lab for r in rows)) for lab in used) + " |")

    notes = {a: {} for a in runs}
    for approach, rows in runs.items():
        for r in rows:
            if r["note"]:
                notes[approach][r["note"]] = notes[approach].get(r["note"], 0) + 1
    lines += ["", "What each approach did (fallbacks, ties, unreadable votes):", ""]
    for approach, counts in notes.items():
        if counts:
            lines.append(f"- {approach}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    text = "\n".join(lines) + "\n"
    (RESULTS / "comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
