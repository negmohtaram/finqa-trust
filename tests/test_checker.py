"""Twenty hand-picked cases from the FinQA and TAT-QA dev sets."""
import pytest

from finqa_trust.checker import accuracy, is_correct, parse_number
from finqa_trust.schema import Record


def rec(gold, answer_type="arithmetic", scale="", source="finqa"):
    return Record(id="case", source=source, split="dev", question="q", text=[], table=[],
                  gold_answer=gold, answer_type=answer_type, scale=scale)


CASES = [
    # (name, record, prediction, expected at 0.5%)
    ("trailing zero", rec(127.4), "127.40", True),
    ("0.7% off", rec(127.4), 126.5, False),
    ("percent sign vs fraction", rec(0.935, scale="percent"), "93.5%", True),
    ("bare percent vs fraction", rec(0.935, scale="percent"), 93.5, True),
    ("no x100 outside percent", rec(127.4), 1.274, False),
    ("negative percent, rounded", rec(-0.52485, scale="percent"), "-52.5%", True),
    ("sign wrong", rec(-0.52485, scale="percent"), 52.49, False),
    ("accounting negative", rec(-1.9), "(1.9)", True),
    ("thousands separators", rec(4575515.0), "4,575,515", True),
    ("wrong cell, large number", rec(778000.0), 790000, False),
    ("zero for a tiny gold", rec(0.00001), 0, False),
    ("honest rounding of a small percent", rec(0.00455, scale="percent"), "0.46%", True),
    ("TAT-QA percent as fraction", rec(-22.22, scale="percent", source="tatqa"), -0.2222, True),
    ("scale word", rec(-12.6, scale="million", source="tatqa"), "-12.6 million", True),
    ("scale written out", rec(-12.6, scale="million", source="tatqa"), -12600000, True),
    ("numeric span", rec("$1,496.5", "span", "million", "tatqa"), "1496.5", True),
    ("multi-span order", rec(["2019", "2018", "2017"], "multi-span", source="tatqa"), ["2017", "2018", "2019"], True),
    ("multi-span missing one", rec(["2019", "2018", "2017"], "multi-span", source="tatqa"), ["2019", "2018"], False),
    ("count as text", rec(4.0, "count", source="tatqa"), "4", True),
    ("yes-no punctuation", rec("no", "yes-no"), "No.", True),
]


@pytest.mark.parametrize("name,record,pred,expected", CASES, ids=[c[0] for c in CASES])
def test_hand_picked_cases(name, record, pred, expected):
    assert is_correct(pred, record) is expected


def test_tolerance_matters_only_where_it_should():
    r = rec(127.4)
    assert is_correct(126.5, r, rel_tol=0.005) is False
    assert is_correct(126.5, r, rel_tol=0.01) is True


def test_rounding_rule_needs_two_significant_digits():
    assert is_correct("13", rec(12.5)) is True         # whole-number rounding accepted (decision D)
    assert is_correct("59%", rec(0.59356, scale="percent")) is True
    assert is_correct("1", rec(1.4)) is False          # one significant digit is too coarse
    assert is_correct("0.5", rec(0.46)) is False


def test_accuracy_reports_three_tolerances_and_counts_missing_as_wrong():
    records = [rec(127.4).model_copy(update={"id": "a"}), rec(10.0).model_copy(update={"id": "b"})]
    result = accuracy({"a": 126.5}, records)
    assert result == {0.001: 0.0, 0.005: 0.0, 0.01: 0.5}


@pytest.mark.parametrize("raw,value,mult", [
    ("$1,496.5", 1496.5, 1), ("(1.9)", -1.9, 1), ("-12.6 million", -12.6, 1e6),
    ("93.5%", 93.5, 1), ("2.3bn", 2.3, 1e9), ("No.", None, None),
    (1e-05, 1e-05, 1), ("4.9\\n", 4.9, 1),
])
def test_parse_number(raw, value, mult):
    n = parse_number(raw)
    if value is None:
        assert n is None
    else:
        assert (n.value, n.multiplier) == (value, mult)


def test_every_fixture_gold_matches_itself():
    from pathlib import Path
    from finqa_trust.data import load
    fixtures = Path(__file__).parent / "fixtures"
    for r in load("finqa", "dev", fixtures) + load("tatqa", "dev", fixtures):
        assert is_correct(r.gold_answer, r), r.id
