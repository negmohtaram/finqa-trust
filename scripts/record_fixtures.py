"""Record real model answers for the fixture questions, once, into tests/fixtures/model/.

Needs GROQ_API_KEY in the environment. Tests replay these files offline.
Usage: python scripts/record_fixtures.py [--model openai/gpt-oss-120b]
"""
import argparse
from pathlib import Path

from finqa_trust.data import load
from finqa_trust.model import DEFAULT_MODEL, GroqClient, ModelRequest, RecordingClient
from finqa_trust.prompts import direct_messages

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    records = (load("finqa", "dev", FIXTURES) + load("tatqa", "dev", FIXTURES))[: args.n]
    client = RecordingClient(GroqClient(), FIXTURES / "model")
    for r in records:
        response = client.complete(ModelRequest(model=args.model, messages=direct_messages(r),
                                                 reasoning_effort="low"))
        print(f"{r.id[:40]:40}  gold={r.gold_answer!s:12.12}  model={response.text.strip()[:30]!r}"
              f"  {response.latency_s:.1f}s  {response.input_tokens}+{response.output_tokens} tokens")
    print(f"recorded {len(client.recorded)} answers in {client.folder}")


if __name__ == "__main__":
    main()
