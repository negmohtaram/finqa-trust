# finqa-trust

Which steps of financial question answering can a language model be trusted with: finding the right numbers in a report, calculating with them, or neither? And what does each common fix (temperature 0, majority voting, letting the model write code) buy in accuracy and cost?

## Data
Two public benchmarks built from real company reports, both with checkable answers:
- [FinQA](https://github.com/czyssrs/FinQA) (Chen et al., 2021): questions over earnings reports, each with the gold calculation program.
- [TAT-QA](https://github.com/NExTplusplus/TAT-QA) (Zhu et al., 2021): questions over tables and text from annual reports.

Only the dev and test splits are used. Test results are reported once, at the end.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py
pytest -q
```

## Baseline: where the model fails
Model: `openai/gpt-oss-120b` (reasoning effort low, temperature 1.0), asked for the numbers it used, its formula and its answer as JSON. 100 arithmetic questions (55 FinQA, 45 TAT-QA) from a fixed stratified dev sample, each asked twice. Every wrong answer is labelled by the first step that broke, checked against the gold calculation:

| Label | Meaning | Share (mean of 2 runs) | Run spread | 95% CI (run 0, bootstrap) |
|---|---|---|---|---|
| correct | matches gold at 0.5% | **77.5%** | 77–78% | 69–85% |
| extraction | a number the gold calculation needs is missing | 9.0% | 8–10% | 4–16% |
| formula | right numbers, logic not equivalent to gold | 10.0% | 10–10% | 4–16% |
| sign | right magnitude, opposite sign | 3.0% | 3–3% | 0–6% |
| arithmetic | right formula, answer is not what it computes to | 0.5% | 0–1% | |
| unreadable / format / unchecked | | 0% | | |

What this says:
- **The model reads and reasons badly far more often than it calculates badly.** Extraction and formula errors are about 19% of answers; arithmetic slips are 0.5%.
- **Repeating a run shows little model randomness:** 95 of 100 questions got the same verdict both times. The much larger uncertainty is which questions were sampled: on 100 questions one accuracy is about ±8 points.
- **FinQA is much harder than TAT-QA here:** 64% vs 94% correct. Typical FinQA failures are choosing the wrong total, ignoring a "minus 100" in cumulative return questions, adding items the question did not ask for, and stopping before the final division.
- **Sign is a convention problem, not a reasoning one.** For questions about a decline the model often gives the positive size. These are scored as wrong and counted separately.

How formulas are compared: a formula that matches the gold number once could be a coincidence, so both formulas are recomputed with every gold number swapped for random values, five times. They count as equivalent only if they agree every time. Gold calculations are used only when they reproduce the gold answer; a few do not (for example a TAT-QA derivation of -114 - (71) with a gold answer of -43), and those items are reported as unchecked. A few golds are simply wrong (an equity share given as 1 - 45.6), so no system reaches 100%.

Reproduce:
```bash
python scripts/make_samples.py
python scripts/run_baseline.py --runs 2      # replays recorded calls when present
```

## How many questions are enough?
On one fixed sample, two approaches can be compared question by question. Only the questions where they disagree carry information, which makes paired comparisons much cheaper than comparing two separate test sets:

| Goal (accuracy about 60%, 95% confidence, 80% power) | Questions needed |
|---|---|
| One accuracy within ±3 points | about 1,025 |
| A 3-point difference, two separate question sets | about 4,186 each |
| A 3-point difference, same questions, 15% disagreement | about 1,308 |
| Smallest difference 400 shared questions can detect (15% disagreement) | about 5.4 points |

![interval width against number of questions](docs/ci_width.png)

Code: `src/finqa_trust/stats.py`.
