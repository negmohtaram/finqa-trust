import json

from finqa_trust.approaches import code_exec, vote


def reply(formula, answer, values=(6823, 6161)):
    return json.dumps({"values": [{"name": "", "value": v} for v in values],
                       "formula": formula, "answer": answer})


def answer_of(final):
    return json.loads(final.reply)["answer"]


def test_code_exec_replaces_a_wrong_mental_result():
    final = code_exec(reply("(6823 - 6161) / 6161 * 100", 11.2))
    assert final.note == "computed"
    assert abs(answer_of(final) - 10.745) < 0.001


def test_code_exec_keeps_the_sign_the_formula_has():
    final = code_exec(reply("(6161 - 6823) / 6823 * 100", 9.7))
    assert answer_of(final) < 0


def test_code_exec_falls_back_visibly_when_the_formula_does_not_compute():
    final = code_exec(reply("growth of revenue", 10.7))
    assert final.note == "fallback"
    assert answer_of(final) == 10.7


def test_vote_takes_the_biggest_group_within_half_a_percent():
    replies = [reply("a", 10.745), reply("b", 52.1), reply("c", 10.75), reply("d", -10.745), reply("e", 10.74)]
    final = vote(replies)
    assert final.note == "3/5"
    assert final.reply == replies[0]


def test_vote_tie_goes_to_the_group_whose_formula_computes_to_its_answer():
    good = "(6823 - 6161) / 6161 * 100"
    replies = [reply("6823 + 6161", 52.1), reply("6823 + 6161", 52.1),
               reply(good, 10.745), reply(good, 10.745), reply("x", 3.0)]
    final = vote(replies)
    assert final.note == "2/5 tie, broken by formula"
    assert final.reply == replies[2]


def test_vote_tie_with_no_way_to_choose_takes_the_earliest_group():
    replies = [reply("x", 1.0), reply("y", 2.0), reply("y", 2.0), reply("x", 1.0)]
    final = vote(replies)
    assert final.note == "2/4 tie, earliest"
    assert final.reply == replies[0]


def test_unreadable_replies_do_not_vote_and_are_counted():
    replies = ["not json", "still not json", reply("a", 7.0)]
    assert vote(replies).note == "1/3, 2 unreadable"
