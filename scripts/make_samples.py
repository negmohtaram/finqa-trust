"""Write the fixed dev samples once: 400 questions for comparing approaches, and the
100 arithmetic questions among them used for repeated baseline runs.

Usage: python scripts/make_samples.py
"""
from collections import Counter

from finqa_trust.data import load_all
from finqa_trust.sample import SAMPLES, save_ids, stratified_sample

SEED = 0


def main() -> None:
    dev = load_all("dev")
    dev400 = stratified_sample(dev, 400, SEED)
    arith100 = stratified_sample([r for r in dev400 if r.answer_type == "arithmetic"], 100, SEED)
    save_ids(dev400, SAMPLES / "dev400.json")
    save_ids(arith100, SAMPLES / "dev400_arith100.json")
    for name, rs in (("dev400", dev400), ("dev400_arith100", arith100)):
        print(name, dict(Counter((r.source, r.answer_type) for r in rs)),
              "ambiguous_gold:", sum(r.ambiguous_gold for r in rs))


if __name__ == "__main__":
    main()
