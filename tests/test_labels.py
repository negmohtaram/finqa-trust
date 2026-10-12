import json

import pytest

from finqa_trust.labels import formulas_equivalent, gold_expression, gold_operands, label, step_table
from finqa_trust.schema import Record


def record(program="subtract(6823, 6161), divide(#0, 6161)", gold=0.10745, scale="percent",
           source="finqa"):
    return Record(id="x", source=source, split="dev", question="What was the change?", text=[],
                  table=[], gold_answer=gold, answer_type="arithmetic", scale=scale, program=program)


def reply(values, formula, answer):
    return json.dumps({"values": [{"name": "", "value": v} for v in values],
                       "formula": formula, "answer": answer})


@pytest.mark.parametrize("values, formula, answer, expected", [
    ([6823, 6161], "(6823 - 6161) / 6161 * 100", "10.74", "correct"),
    ([6823, 6123], "(6823 - 6123) / 6123 * 100", "11.43", "extraction"),
    ([6823, 6161], "(6823 - 6161) / 6823 * 100", "9.70", "formula"),
    ([6823, 6161], "(6823 - 6161) / 6161 * 100", "12.1", "arithmetic"),
    ([6823, 6161], "(6823 - 6161) / 6161 * 100", "increased by 10.7 percent", "format"),
])
def test_first_broken_step(values, formula, answer, expected):
    assert label(reply(values, formula, answer), record())["label"] == expected


def test_both_formula_and_arithmetic_wrong_is_formula():
    assert label(reply([6823, 6161], "(6823 - 6161) / 6823 * 100", "12.1"), record())["label"] == "formula"


def test_unreadable_reply():
    assert label("10.74", record())["label"] == "unreadable"


def test_equivalent_formula_written_differently():
    out = label(reply([6823, 6161], "(6823 / 6161 - 1) * 100", "10.75"), record())
    assert out["label"] == "correct" and out["equivalent"] is True and not out["lucky"]


def test_right_answer_from_wrong_formula_is_lucky():
    # 662 / 6161 happens to be the gold, but 662 is not a gold operand: a coincidence, not logic
    out = label(reply([6823, 6161, 662], "(6823 - 6161) / 6823 * 100 * 1.10745", "10.745"), record())
    assert out["label"] == "correct" and out["lucky"] is True


def test_swapped_inputs_catch_a_coincidence():
    assert formulas_equivalent("(6823 - 6161) / 6823 * 1.10745", "(6823 - 6161) / 6161",
                               [6161.0, 6823.0]) is False


def test_percent_operands_and_values_match_as_fractions():
    r = record(program="multiply(526.5, 2.4%)", gold=12.636, scale="")
    assert label(reply([526.5, 0.024], "526.5 * 0.024", "12.64"), r)["label"] == "correct"
    assert label(reply([526.5, 2.4], "526.5 * 2.4 / 100", "12.0"), r)["label"] == "arithmetic"


def test_tatqa_derivation_with_dollars_and_brackets():
    r = record(program="[(166+178)/2] - [(57+44)/2]", gold=121.5, scale="", source="tatqa")
    assert gold_expression(r) is not None
    assert gold_operands(r) == [2.0, 44.0, 57.0, 166.0, 178.0]


def test_gold_that_contradicts_its_own_formula_is_unchecked():
    r = record(program="-114 - (71)", gold=-43.0, scale="", source="tatqa")
    assert gold_expression(r) is None
    assert label(reply([114, 71], "-114 - 71", "-185"), r)["label"] == "unchecked"


def test_table_operations_are_unchecked_not_guessed():
    r = record(program="table_average(revenue, none)", gold=10.0, scale="")
    assert label(reply([9, 11], "(9 + 11) / 2 + 1", "11"), r)["label"] == "unchecked"


def test_step_table_mean_and_spread():
    rows = [{"run": 0, "label": "correct"}, {"run": 0, "label": "formula"},
            {"run": 1, "label": "correct"}, {"run": 1, "label": "correct"}]
    by_label = {row["label"]: row for row in step_table(rows)}
    assert by_label["correct"]["mean"] == pytest.approx(0.75)
    assert by_label["correct"]["min"] == 0.5 and by_label["correct"]["max"] == 1.0
    assert by_label["formula"]["sd"] == pytest.approx(0.5 ** 0.5 * 0.5)


def test_opposite_sign_is_its_own_label():
    r = record(program="subtract(47.0, 70.1), divide(#0, 70.1)", gold=-0.32953)
    assert label(reply([70.1, 47.0], "(70.1 - 47.0) / 70.1 * 100", "32.95"), r)["label"] == "sign"


def test_not_lucky_when_gold_uses_a_number_the_model_derived():
    # gold divides a precomputed difference (-380); the model computes it from 891 and 1271
    r = record(program="divide(-380, 1271)", gold=-0.29898)
    out = label(reply([1271, 891], "(891 - 1271) / 1271 * 100", "-29.9"), r)
    assert out["label"] == "correct" and not out["lucky"]


def test_hundred_as_a_gold_number_is_not_swapped():
    r = record(program="subtract(104.11, 100.00), divide(#0, 100.00)", gold=0.0411)
    out = label(reply([100.0, 104.11], "(104.11 - 100.00) / 100.00 * 100", "4.11"), r)
    assert out["label"] == "correct" and out["equivalent"] is True and not out["lucky"]
