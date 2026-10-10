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

## Status
Loader and answer checker in progress. Results will be added here.
