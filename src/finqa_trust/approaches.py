"""Ways to turn model replies into one final reply, so every approach is scored by the same labeler.

code_exec: keep the model's numbers and formula, but take the answer from computing the formula.
vote: several replies to the same question; the most common answer wins.
"""
import json
from dataclasses import dataclass

from .checker import parse_number
from .labels import ANSWER_TOL, _answer_is_result
from .structured import StructuredAnswer, eval_formula, parse_structured


@dataclass(frozen=True)
class Final:
    reply: str              # a reply in the model's JSON shape, ready for label()
    note: str = ""          # what the approach did, e.g. "computed", "fallback", "tie"


def code_exec(reply: str) -> Final:
    """Replace the answer with the formula's result. If the formula can't be computed,
    keep the model's own answer and say so, so the fallback is counted, not hidden."""
    parsed = parse_structured(reply)
    if parsed is None:
        return Final(reply, "unreadable")
    result = eval_formula(parsed.formula) if parsed.formula else None
    if result is None:
        return Final(reply, "fallback")
    return Final(_with_answer(parsed, round(result, 6)), "computed")


def vote(replies: list[str]) -> Final:
    """Group answers that agree within 0.5%; the biggest group wins.
    Tie: prefer a group whose formula computes to its own answer. Still tied: the earliest group."""
    groups: list[list[tuple[int, StructuredAnswer]]] = []
    for i, reply in enumerate(replies):
        parsed = parse_structured(reply)
        number = parse_number(parsed.answer) if parsed else None
        if number is None:
            continue
        for group in groups:
            if _same(number.value, parse_number(group[0][1].answer).value):
                group.append((i, parsed))
                break
        else:
            groups.append([(i, parsed)])
    if not groups:
        return Final(replies[0], "no votes")
    top = max(len(g) for g in groups)
    tied = [g for g in groups if len(g) == top]
    note = f"{top}/{len(replies)}"
    unreadable = len(replies) - sum(len(g) for g in groups)
    if unreadable:
        note += f", {unreadable} unreadable"
    if len(tied) > 1:
        consistent = [g for g in tied if any(_consistent(p) for _, p in g)]
        note += " tie, broken by formula" if len(consistent) == 1 else " tie, earliest"
        tied = consistent or tied
    winner = min(tied, key=lambda g: g[0][0])
    return Final(replies[winner[0][0]], note)


def _same(a: float, b: float) -> bool:
    if a == b:
        return True
    return abs(a - b) <= ANSWER_TOL * max(abs(a), abs(b))


def _consistent(parsed: StructuredAnswer) -> bool:
    result = eval_formula(parsed.formula) if parsed.formula else None
    return result is not None and _answer_is_result(parsed.answer, result)


def _with_answer(parsed: StructuredAnswer, answer: float) -> str:
    data = parsed.model_dump()
    data["answer"] = answer
    return json.dumps(data)
