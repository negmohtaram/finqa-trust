"""Run the baseline (model alone, structured answer) several times on the same questions,
label every answer by the step that broke, and print the per-step table.

Needs GROQ_API_KEY for answers not yet recorded. Rerunning replays finished calls for free.
Usage: python scripts/run_baseline.py [--runs 5] [--temperature 1.0] [--pause 2.0]
"""
import argparse
import json
import time
from pathlib import Path

from finqa_trust.data import load_all
from finqa_trust.labels import label, step_table
from finqa_trust.model import DEFAULT_MODEL, GroqClient, ModelRequest, ResumingClient
from finqa_trust.prompts import structured_messages
from finqa_trust.sample import SAMPLES, load_ids

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def ask(client, request, pause):
    """Call with a pause between live calls; wait and retry on rate limits."""
    for attempt in range(6):
        before = client.live_calls
        try:
            response = client.complete(request)
        except RuntimeError as err:
            if "429" not in str(err) or attempt == 5:
                raise
            time.sleep(30 * (attempt + 1))
            continue
        if client.live_calls > before:
            time.sleep(pause)
        return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample", default="dev400_arith100.json")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between live calls")
    args = parser.parse_args()

    records = load_ids(SAMPLES / args.sample, load_all("dev"))
    client = ResumingClient(GroqClient(), RESULTS / "recordings")
    rows = []
    for run in range(args.runs):
        for i, r in enumerate(records):
            request = ModelRequest(model=args.model, messages=structured_messages(r),
                                   temperature=args.temperature, reasoning_effort="low",
                                   json_mode=True, run=run)
            response = ask(client, request, args.pause)
            rows.append({"id": r.id, "source": r.source, "run": run, "reply": response.text,
                         "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                         **label(response.text, r)})
            print(f"run {run} item {i + 1:3}/{len(records)}  {rows[-1]['label']:10}  live calls {client.live_calls}",
                  end="\r")
        print()

    out = RESULTS / f"baseline_runs_t{args.temperature}.jsonl"
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"\n{len(rows)} answers -> {out}  ({client.live_calls} live calls)\n")
    print(f"{'label':12}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}")
    for row in step_table(rows):
        print(f"{row['label']:12}{row['mean']:8.1%}{row['sd']:8.1%}{row['min']:8.1%}{row['max']:8.1%}")


if __name__ == "__main__":
    main()
