import pytest

from finqa_trust.structured import eval_formula, parse_structured


def test_parse_plain_and_fenced():
    reply = '{"values": [{"name": "rev 2019", "value": 5829}], "formula": "5829 - 5735", "answer": "94"}'
    for text in (reply, f"```json\n{reply}\n```"):
        parsed = parse_structured(text)
        assert parsed.values[0].value == 5829 and parsed.formula == "5829 - 5735"


@pytest.mark.parametrize("text", ["94", "", '{"values": []}', '{"answer": "1", "values": [{"value": "n/a"}]}'])
def test_unreadable_replies_give_none(text):
    assert parse_structured(text) is None


@pytest.mark.parametrize("formula, expected", [
    ("(5829 - 5735) / 5735 * 100", (5829 - 5735) / 5735 * 100),
    ("-12.6 + 1,000", 987.4),
    ("44.1-56.7", 44.1 - 56.7),
])
def test_eval_formula(formula, expected):
    assert eval_formula(formula) == pytest.approx(expected)


@pytest.mark.parametrize("formula", ["__import__('os')", "2 ** 10", "x + 1", "1 / 0", "", "sum([1, 2])"])
def test_eval_formula_refuses_anything_else(formula):
    assert eval_formula(formula) is None


def test_values_written_as_in_the_report():
    reply = ('{"values": [{"name": "a", "value": "4,206"}, {"name": "b", "value": "$668"}, '
             '{"name": "c", "value": "61%"}, {"name": "d", "value": "(1.9)"}], "formula": "", "answer": 1}')
    assert [v.value for v in parse_structured(reply).values] == [4206, 668, 61, -1.9]


def test_abs_is_allowed_but_no_other_function():
    assert eval_formula("abs(27.4 - 33.5)") == pytest.approx(6.1)
    assert eval_formula("abs(1, 2)") is None
    assert eval_formula("max(1, 2)") is None
