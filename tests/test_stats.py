import random

import pytest

from finqa_trust.stats import bootstrap_ci, ci_half_width, n_for_difference, paired_bootstrap_diff


def flags(p, n, seed=0):
    rng = random.Random(seed)
    return [rng.random() < p for _ in range(n)]


def test_bootstrap_matches_the_formula_on_large_samples():
    f = flags(0.6, 1000)
    acc, lo, hi = bootstrap_ci(f)
    assert lo < acc < hi
    assert (hi - lo) / 2 == pytest.approx(ci_half_width(acc, 1000), rel=0.15)


def test_interval_shrinks_with_more_questions():
    widths = [bootstrap_ci(flags(0.6, n))[2] - bootstrap_ci(flags(0.6, n))[1] for n in (100, 400, 1600)]
    assert widths[0] > widths[1] > widths[2]
    assert widths[0] / widths[2] == pytest.approx(4, rel=0.3)      # 16x the questions, 1/4 the width


def test_all_correct_gives_a_zero_width_interval():
    assert bootstrap_ci([1] * 50) == (1.0, 1.0, 1.0)


def test_same_seed_same_interval():
    f = flags(0.6, 100)
    assert bootstrap_ci(f, seed=3) == bootstrap_ci(f, seed=3)


def test_pairing_beats_ignoring_it():
    # B fixes 6 of A's wrong answers and breaks 2 of its right ones: +4 points, few disagreements
    a = flags(0.6, 200, seed=1)
    b = list(a)
    wrong = [i for i, x in enumerate(a) if not x][:6]
    right = [i for i, x in enumerate(a) if x][:2]
    for i in wrong:
        b[i] = True
    for i in right:
        b[i] = False
    diff, lo, hi = paired_bootstrap_diff(a, b)
    assert diff == pytest.approx(0.02)
    unpaired_half = (ci_half_width(sum(a) / 200, 200) ** 2 + ci_half_width(sum(b) / 200, 200) ** 2) ** 0.5
    assert (hi - lo) / 2 < unpaired_half / 2


def test_paired_needs_unequal_lengths_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 0], [1])


def test_sample_size_for_three_points():
    assert n_for_difference(0.03, accuracy=0.6) == pytest.approx(4186, abs=5)
    assert n_for_difference(0.03, discordant=0.15) == pytest.approx(1308, abs=5)
