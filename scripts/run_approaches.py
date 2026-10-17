"""Run one approach on the fixed sample, label every final answer and save one row per question.

  alone   one answer at temperature 1.0 (the baseline's run 0)
  code    the same answer, with the result computed from the model's formula
  temp0   one answer at temperature 0
  vote    k answers at temperature 1.0 (runs 0..k-1, so the baseline runs are reused), most common wins

Calls already recorded replay for free. When the daily quota runs out the script stops,
keeps what it recorded and can be rerun the next day.
Usage: python scripts/run_approaches.py --approach vote [--k 5] [--max-live-calls 90]
"""
import argparse
import json
import sys
from pathlib import Path

from finqa_trust.approaches import Final, code_exec, vote
from finqa_trust.data import load_all
from finqa_trust.labels import label
from finqa_trust.model import DEFAULT_MODEL, DailyLimit, GroqClient, ModelRequest, ResumingClient, ask
from finqa_trust.prompts import structured_messages
from finqa_trust.sample import SAMPLES, load_ids

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
APPROACHES = ("alone", "code", "temp0", "vote")


def requests_for(approach: str, record, model: str, k: int) -> list[ModelRequest]:
    temperature = 0.0 if approach == "temp0" else 1.0
    runs = range(k) if approach == "vote" else range(1)
    return [ModelRequest(model=model, messages=structured_messages(record), temperature=temperature,
                         reasoning_effort="low", json_mode=True, run=run) for run in runs]


def final_for(approach: str, texts: list[str]) -> Final:
    if approach == "code":
        return code_exec(texts[0])
    if approach == "vote":
        return vote(texts)
    return Final(texts[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", choices=APPROACHES, required=True)
    parser.add_argument("--k", type=int, default=5, help="votes per question")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", default="dev400_arith100.json")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between live calls")
    parser.add_argument("--max-live-calls", type=int, default=None, help="stop after this many new calls")
    args = parser.parse_args()

    records = load_ids(SAMPLES / args.sample, load_all("dev"))
    client = ResumingClient(GroqClient(), RESULTS / "recordings")
    rows = []
    for i, r in enumerate(records):
        responses = []
        for request in requests_for(args.approach, r, args.model, args.k):
            if (args.max_live_calls is not None and client.live_calls >= args.max_live_calls
                    and not client.is_recorded(request)):
                sys.exit(f"\nStopped at {client.live_calls} live calls (item {i + 1}). Rerun to continue.")
            try:
                responses.append(ask(client, request, args.pause))
            except DailyLimit:
                sys.exit(f"\nDaily quota used up after {client.live_calls} live calls (item {i + 1}). "
                         "Rerun tomorrow to continue.")
        final = final_for(args.approach, [x.text for x in responses])
        latencies = [x.latency_s for x in responses]
        rows.append({"id": r.id, "source": r.source, "approach": args.approach, "note": final.note,
                     "calls": len(responses),
                     "input_tokens": sum(x.input_tokens for x in responses),
                     "output_tokens": sum(x.output_tokens for x in responses),
                     "latency_serial_s": round(sum(latencies), 3),
                     "latency_parallel_s": max(latencies),
                     "reply": final.reply, **label(final.reply, r)})
        if (i + 1) % 10 == 0:
            print(f"item {i + 1:3}/{len(records)}  live calls {client.live_calls}", flush=True)

    suffix = f"_k{args.k}" if args.approach == "vote" else ""
    out = RESULTS / f"{args.approach}{suffix}.jsonl"
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    correct = sum(row["correct"] for row in rows)
    print(f"\n{args.approach}: {correct}/{len(rows)} correct -> {out}  ({client.live_calls} live calls)")


if __name__ == "__main__":
    main()
