"""Decide whether a model's answer matches the gold answer of a Record.

Rules:
- Numbers match within a relative tolerance (default 0.5%). An absolute
  tolerance is used only when the gold answer is exactly 0.
- Honest rounding is accepted: if the model wrote d decimals (d may be 0) and at
  least 2 significant digits, the gold rounded to d decimals may equal the
  prediction ("59%" for 59.356%, "0.46%" for 0.455%).
- Percent questions only: 12.5 and 0.125 are the same answer.
- Scaled golds (thousand / million / billion): -12.6 million matches -12,600,000.
- Text answers: case and surrounding punctuation are ignored; multi-span
  answers are compared as sets; spans that are numbers are compared as numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from .schema import Record

TOLERANCES = (0.001, 0.005, 0.01)
ZERO_ABS_TOL = 1e-4
SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
_UNIT_WORDS = {"thousand": 1e3, "thousands": 1e3, "k": 1e3, "million": 1e6, "millions": 1e6,
               "m": 1e6, "mn": 1e6, "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9}
_NUMBER = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)$")


@dataclass(frozen=True)
class ParsedNumber:
    value: float          # the number as written, e.g. 12.6 for "-12.6 million" -> -12.6
    multiplier: float     # 1e6 for "million", 1 otherwise
    decimals: int         # digits after the decimal point, as written
    sig_digits: int       # significant digits, as written


def parse_number(raw: object) -> ParsedNumber | None:
    """Read a number from model output: '$1,496.5', '(1.9)', '-12.6 million', '93.5%'."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        text = format(Decimal(repr(raw)), "f")                 # 1e-05 -> "0.00001", never e-notation
    elif isinstance(raw, str):
        text = raw
    else:
        return None
    text = text.replace("\\n", " ").strip().lower().rstrip(".").strip()   # literal "\n" left in some data
    negative = False
    if text.startswith("(") and text.endswith(")"):          # accounting notation
        negative, text = True, text[1:-1].strip()
    text = re.sub(r"[$€£¥]|usd|eur|sar", "", text).replace(",", "").replace(" ", "")
    text = text.rstrip("%")
    multiplier = 1.0
    for word in sorted(_UNIT_WORDS, key=len, reverse=True):
        if text.endswith(word) and _NUMBER.match(text[: -len(word)] or "x"):
            multiplier, text = _UNIT_WORDS[word], text[: -len(word)]
            break
    if not _NUMBER.match(text):
        return None
    digits = text.lstrip("+-")
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    sig = len(digits.replace(".", "").lstrip("0"))
    value = float(text)
    return ParsedNumber(-value if negative else value, multiplier, decimals, sig)


def _clean(x: float) -> float:
    """Remove float noise such as 0.00455 * 100 = 0.45499999999999996."""
    return float(f"{x:.12g}")


def _close(pred: float, gold: float, rel_tol: float) -> bool:
    if gold == 0:
        return abs(pred) <= ZERO_ABS_TOL
    return abs(pred - gold) <= rel_tol * abs(gold)


def _rounds_to(pred: ParsedNumber, pred_value: float, gold: float, min_sig: int = 2) -> bool:
    if pred.sig_digits < min_sig:
        return False
    quantum = Decimal(1).scaleb(-pred.decimals)
    rounded = Decimal(repr(_clean(gold))).quantize(quantum, rounding=ROUND_HALF_UP)
    return rounded == Decimal(repr(_clean(pred_value))).quantize(quantum, rounding=ROUND_HALF_UP)


def _gold_readings(gold: float, record: Record) -> list[float]:
    readings = [gold]
    if record.scale in SCALES:
        readings.append(gold * SCALES[record.scale])
    if record.scale == "percent":
        readings += [gold * 100, gold / 100]
    return [_clean(g) for g in readings]


def numbers_match(pred_raw: object, gold: float, record: Record, rel_tol: float,
                  min_sig: int = 2) -> bool:
    pred = parse_number(pred_raw)
    if pred is None:
        return False
    pred_readings = {_clean(pred.value), _clean(pred.value * pred.multiplier)}
    for g in _gold_readings(gold, record):
        for p in pred_readings:
            if _close(p, g, rel_tol):
                return True
            if p == _clean(pred.value) and _rounds_to(pred, p, g, min_sig):
                return True
    return False


def agrees_up_to_rounding(text_answer: object, record: Record) -> bool:
    """True if a written answer is the gold, possibly rounded (even to 1 significant digit).
    Used to audit dataset answers, not to score models."""
    return numbers_match(text_answer, float(record.gold_answer), record, rel_tol=0.01, min_sig=1)


def normalize_text(s: object) -> str:
    text = str(s).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .\"'")


def _split_spans(pred: object) -> list[str]:
    if isinstance(pred, (list, tuple, set)):
        return [str(p) for p in pred]
    return [p for p in re.split(r"\s*(?:,|;|\band\b)\s*", str(pred)) if p.strip()]


def _span_key(s: str) -> str:
    n = parse_number(s)
    return repr(_clean(n.value * n.multiplier)) if n else normalize_text(s)


def is_correct(pred: object, record: Record, rel_tol: float = 0.005) -> bool:
    gold = record.gold_answer
    if record.answer_type in ("arithmetic", "count"):
        return numbers_match(pred, float(gold), record, rel_tol)
    if record.answer_type == "multi-span":
        return {_span_key(p) for p in _split_spans(pred)} == {_span_key(g) for g in gold}
    # span and yes-no
    gold_number = parse_number(gold)
    if gold_number is not None and parse_number(pred) is not None:
        return numbers_match(pred, gold_number.value * gold_number.multiplier, record, rel_tol)
    return normalize_text(pred) == normalize_text(gold)


def accuracy(preds: dict[str, object], records: list[Record]) -> dict[float, float]:
    """Accuracy at each tolerance in TOLERANCES. Missing predictions count as wrong."""
    return {tol: sum(is_correct(preds.get(r.id), r, tol) for r in records) / len(records)
            for tol in TOLERANCES}


def report(preds: dict[str, object], records: list[Record]) -> dict:
    """Accuracy on all records and on records whose gold answer is unambiguous."""
    clean = [r for r in records if not r.ambiguous_gold]
    return {"all": accuracy(preds, records), "clean": accuracy(preds, clean),
            "n_all": len(records), "n_clean": len(clean)}
