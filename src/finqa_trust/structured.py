"""Read the model's structured reply: the numbers it took, its formula and its answer."""
import ast
import json
import operator

from pydantic import BaseModel, ValidationError


class Value(BaseModel):
    name: str = ""
    value: float


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


def eval_formula(formula: str) -> float | None:
    """Compute + - * / and parentheses on numbers only. Anything else, or dividing by zero,
    gives None. Never uses eval(): the formula is text written by a model."""
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError("not plain arithmetic")
    try:
        return walk(ast.parse(formula.replace(",", ""), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None
