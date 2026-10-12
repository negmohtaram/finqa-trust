"""Read the model's structured reply: the numbers it took, its formula and its answer."""
import ast
import json
import operator
from typing import Callable

from pydantic import BaseModel, ValidationError, field_validator

from .checker import parse_number


class Value(BaseModel):
    name: str = ""
    value: float

    @field_validator("value", mode="before")
    @classmethod
    def read_written_number(cls, v):
        """Models copy numbers as written: "4,206", "$668", "61%", "(1.9)"."""
        if isinstance(v, str):
            parsed = parse_number(v)
            if parsed is None:
                raise ValueError(f"not a number: {v!r}")
            return parsed.value
        return v


class StructuredAnswer(BaseModel):
    values: list[Value] = []
    formula: str = ""
    answer: float | str | list[str]


def parse_structured(text: str) -> StructuredAnswer | None:
    """Parse the reply, tolerating a code fence around it. None means unreadable."""
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`").removeprefix("json").strip()
    try:
        return StructuredAnswer.model_validate(json.loads(body))
    except (json.JSONDecodeError, ValidationError):
        return None


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv}


def eval_formula(formula: str, swap: Callable[[float], float] | None = None) -> float | None:
    """Compute + - * / and parentheses on numbers only. Anything else, or dividing by zero,
    gives None. Never uses eval(): the formula is text written by a model.
    `swap`, if given, replaces each number before computing (used to test formulas on new inputs)."""
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return swap(float(node.value)) if swap else float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "abs"
                and len(node.args) == 1 and not node.keywords):
            return abs(walk(node.args[0]))
        raise ValueError("not plain arithmetic")
    try:
        text = formula.replace(",", "").replace("[", "(").replace("]", ")").replace("%", "").replace("$", "")
        return walk(ast.parse(text, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None
