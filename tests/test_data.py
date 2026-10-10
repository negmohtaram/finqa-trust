from pathlib import Path

import pytest
from pydantic import ValidationError

from finqa_trust.data import load
from finqa_trust.schema import Record

FIXTURES = Path(__file__).parent / "fixtures"


def test_finqa_fixture_loads_into_records():
    records = load("finqa", "dev", FIXTURES)
    assert records and all(isinstance(r, Record) for r in records)
    assert all(r.source == "finqa" and r.program for r in records)


def test_finqa_percent_answers_are_marked_percent():
    records = load("finqa", "dev", FIXTURES)
    percent = [r for r in records if r.scale == "percent"]
    assert percent, "fixture should contain a percentage question"
    assert all(abs(r.gold_answer) < 10 for r in percent)   # stored as a fraction, e.g. 0.125


def test_tatqa_fixture_covers_every_answer_type():
    records = load("tatqa", "dev", FIXTURES)
    assert {r.answer_type for r in records} >= {"span", "multi-span", "arithmetic", "count"}
    for r in records:
        if r.answer_type in ("arithmetic", "count"):
            assert isinstance(r.gold_answer, float)
        if r.answer_type == "multi-span":
            assert isinstance(r.gold_answer, list)


def test_record_rejects_unknown_fields_and_empty_questions():
    base = dict(id="x", source="finqa", split="dev", question="q", text=[], table=[],
                gold_answer=1.0, answer_type="arithmetic")
    with pytest.raises(ValidationError):
        Record(**base | {"surprise": 1})
    with pytest.raises(ValidationError):
        Record(**base | {"question": "  "})
    with pytest.raises(ValidationError):
        Record(**base | {"source": "other"})


@pytest.mark.skipif(not (Path(__file__).parents[1] / "data/raw/finqa_dev.json").exists(),
                    reason="full data not downloaded")
def test_full_dev_and_test_sets_load():
    from finqa_trust.data import load_all
    for split in ("dev", "test"):
        records = load_all(split)
        assert len(records) > 2500
        assert len({r.id for r in records}) == len(records)    # ids are unique


def test_finqa_items_with_conflicting_answers_are_flagged():
    from finqa_trust.data import _finqa_record
    item = {"id": "X/1", "pre_text": [], "post_text": [], "table": [["a", "1"]],
            "qa": {"question": "q", "answer": "25%", "exe_ans": 0.19495, "program": "divide(1, 2)"}}
    assert _finqa_record(item, "dev").ambiguous_gold is True
    item["qa"]["answer"] = "19.5%"
    assert _finqa_record(item, "dev").ambiguous_gold is False
    item["qa"]["answer"] = "19%"                     # coarse rounding is not a disagreement
    assert _finqa_record(item, "dev").ambiguous_gold is False


def test_report_gives_all_and_clean_accuracy():
    from finqa_trust.checker import report
    base = dict(source="finqa", split="dev", question="q", text=[], table=[], answer_type="arithmetic")
    records = [Record(id="a", gold_answer=1.0, **base),
               Record(id="b", gold_answer=2.0, ambiguous_gold=True, **base)]
    result = report({"a": 1.0, "b": 9.0}, records)
    assert result["all"][0.005] == 0.5 and result["clean"][0.005] == 1.0
    assert (result["n_all"], result["n_clean"]) == (2, 1)
