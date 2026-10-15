"""Label each answer by the first step that broke: extraction, formula, arithmetic or format.

    unreadable  the reply is not the JSON we asked for
    correct     the answer matches gold (flagged lucky if it used every gold number but its
                formula is not equivalent to gold)
    sign        the answer is the gold with the opposite sign ("32.95" for a decline of -32.95%)
    extraction  a number the gold calculation needs is missing from the model's values
    formula     the values are right but the formula is not equivalent to the gold one
    arithmetic  the formula is right but the written answer is not what it computes to
    format      the formula is right but the answer is not written as a number
    unchecked   wrong, but the gold calculation uses an operation we cannot replay

Formula equivalence is tested on new inputs, not only the original ones: every gold operand
is swapped for a random nearby value in both formulas at once, several times. A wrong formula
can hit the gold number once by coincidence, almost never on fresh numbers.
"""
from __future__ import annotations

import math
import random
import re

from .checker import _clean, is_correct, numbers_match, parse_number
from .schema import Record
from .structured import StructuredAnswer, eval_formula, parse_structured

LABELS = ("correct", "sign", "extraction", "formula", "arithmetic", "format", "unreadable", "unchecked")
VALUE_TOL = 0.001          # a copied number must match the report within 0.1%
ANSWER_TOL = 0.005         # same tolerance the checker uses for answers
SAME_TOL = 1e-6            # two formulas agree on swapped inputs
SCALE_FORMS = (1.0, 100.0, 0.01)   # 12.5 and 0.125 are the same percentage
SWAPS = 5
FIXED = (100.0, 1000.0)    # read as constants, not swapped, even when the gold lists them as numbers

_FINQA_OPS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
_STEP = re.compile(r"([a-z_]+)\(([^()]*)\)")
_LITERAL = re.compile(r"(?<![#\w.])\d[\d,]*(?:\.\d+)?%?")


def _finqa_arg(arg: str, steps: list[str]) -> str:
    arg = arg.strip()
    if arg.startswith("#"):
        return f"({steps[int(arg[1:])]})"
    if arg.startswith("const_"):
        value = arg[len("const_"):]
        return "-1" if value == "m1" else value
    if arg.endswith("%"):
        return f"({arg[:-1]} / 100)"
    return arg


def gold_expression(record: Record) -> str | None:
    """The gold calculation as plain arithmetic. None if it uses another operation, or if it
    does not reproduce the gold answer (the dataset disagrees with itself)."""
    expr = _program_to_arithmetic(record)
    if expr is None:
        return None
    result = eval_formula(expr)
    if result is None or not numbers_match(result, float(record.gold_answer), record, ANSWER_TOL):
        return None
    return expr


def _program_to_arithmetic(record: Record) -> str | None:
    if not record.program:
        return None
    if record.source == "tatqa":
        return record.program.strip()
    steps: list[str] = []
    for op, args in _STEP.findall(record.program):
        if op not in _FINQA_OPS:
            return None
        a, b = (_finqa_arg(x, steps) for x in args.split(","))
        steps.append(f"{a} {_FINQA_OPS[op]} {b}")
    return steps[-1] if steps else None


def gold_operands(record: Record) -> list[float]:
    """Numbers the gold calculation reads from the report: no constants, no step references."""
    if not record.program:
        return []
    values = [float(m.rstrip("%").replace(",", "")) for m in _LITERAL.findall(record.program)]
    return sorted({abs(v) for v in values if v != 0})


def _same_number(a: float, b: float, tol: float) -> bool:
    return any(abs(a * f - b) <= tol * max(abs(b), 1e-12) for f in SCALE_FORMS)


def has_operands(values: list[float], operands: list[float]) -> bool:
    """Every gold operand appears among the model's values (any sign, as % or fraction)."""
    return all(any(_same_number(abs(v), g, VALUE_TOL) for v in values) for g in operands)


def _matching_operand(x: float, operands: list[float]) -> tuple[float, float] | None:
    for g in operands:
        for f in SCALE_FORMS:
            if abs(abs(x) * f - g) <= VALUE_TOL * g:
                return g, f
    return None


def formulas_equivalent(model_formula: str, gold_expr: str, operands: list[float],
                        seed: int = 0) -> bool | None:
    """True if both formulas give the same result (up to a ×100 percent factor) on the original
    operands and on SWAPS random replacements of them. None if either cannot be computed."""
    operands = [g for g in operands if g not in FIXED]
    rng = random.Random(seed)
    trials = [{g: g for g in operands}]
    trials += [{g: g * rng.uniform(0.5, 1.5) for g in operands} for _ in range(SWAPS)]
    ratios = []
    for new in trials:
        def swap(x: float, new=new) -> float:
            match = _matching_operand(x, operands)
            if match is None:
                return x                       # a constant such as 100, left alone
            g, f = match
            return math.copysign(new[g] / f, x)
        m, g = eval_formula(model_formula, swap), eval_formula(gold_expr, swap)
        if m is None or g is None:
            return None
        if g == 0 or m == 0:
            if abs(m - g) > SAME_TOL:
                return False
            continue
        ratios.append(m / g)
    if not ratios:
        return True
    return any(all(abs(r - f) <= SAME_TOL * f for r in ratios) for f in SCALE_FORMS)


def _answer_is_result(answer: object, result: float) -> bool:
    parsed = parse_number(answer)
    return any(_same_number(_clean(v), result, ANSWER_TOL)
               for v in (parsed.value, parsed.value * parsed.multiplier))


def label(reply: str, record: Record, seed: int = 0) -> dict:
    """One label per reply, plus the facts behind it."""
    parsed: StructuredAnswer | None = parse_structured(reply)
    if parsed is None:
        return {"label": "unreadable", "correct": False, "lucky": False, "equivalent": None}
    correct = is_correct(parsed.answer, record)
    gold_expr = gold_expression(record)
    operands = gold_operands(record)
    found = has_operands([v.value for v in parsed.values], operands)
    equivalent = (formulas_equivalent(parsed.formula, gold_expr, operands, seed)
                  if gold_expr and parsed.formula else None)
    # Lucky needs every gold number in the values: when the gold uses a number the model
    # derived itself (a difference or total), the swap test cannot compare the two formulas.
    facts = {"correct": correct, "equivalent": equivalent,
             "lucky": correct and found and equivalent is False}
    if correct:
        return {"label": "correct", **facts}
    number = parse_number(parsed.answer)
    if number is not None and is_correct(-number.value * number.multiplier, record):
        return {"label": "sign", **facts}
    if gold_expr is None:
        return {"label": "unchecked", **facts}
    if not found:
        return {"label": "extraction", **facts}
    if not equivalent:
        return {"label": "formula", **facts}
    if parse_number(parsed.answer) is None:
        return {"label": "format", **facts}
    if not _answer_is_result(parsed.answer, eval_formula(parsed.formula)):
        return {"label": "arithmetic", **facts}
    return {"label": "format", **facts}


def step_table(rows: list[dict]) -> list[dict]:
    """Share of answers per label in each run, then mean and spread across runs.
    rows: one dict per answer with keys run and label (and lucky)."""
    runs = sorted({r["run"] for r in rows})
    table = []
    for name in LABELS + ("lucky",):
        shares = []
        for run in runs:
            in_run = [r for r in rows if r["run"] == run]
            hits = sum(r.get("lucky", False) if name == "lucky" else r["label"] == name for r in in_run)
            shares.append(hits / len(in_run))
        mean = sum(shares) / len(shares)
        spread = (sum((s - mean) ** 2 for s in shares) / (len(shares) - 1)) ** 0.5 if len(shares) > 1 else 0.0
        table.append({"label": name, "mean": mean, "sd": spread, "min": min(shares), "max": max(shares)})
    return table
