from collections import Counter
from pathlib import Path

import pytest

from finqa_trust.data import load_all
from finqa_trust.sample import load_ids, save_ids, stratified_sample

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def records():
    return load_all("dev", FIXTURES)


def mix(rs):
    return Counter((r.source, r.answer_type) for r in rs)


def test_sample_size_and_no_duplicates(records):
    s = stratified_sample(records, 6, seed=0)
    assert len(s) == 6 and len({r.id for r in s}) == 6


def test_sample_keeps_the_mix(records):
    s = stratified_sample(records, len(records) // 2, seed=0)
    full, half = mix(records), mix(s)
    for group, count in full.items():
        assert abs(half[group] - count / 2) <= 1


def test_same_seed_same_sample(records):
    assert [r.id for r in stratified_sample(records, 6, 0)] == [r.id for r in stratified_sample(records, 6, 0)]


def test_saved_sample_round_trip(records, tmp_path):
    s = stratified_sample(records, 6, seed=1)
    save_ids(s, tmp_path / "ids.json")
    assert load_ids(tmp_path / "ids.json", records) == s
