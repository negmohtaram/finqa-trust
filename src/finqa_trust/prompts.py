"""Turn a Record into model messages."""
from .schema import Record

SYSTEM = ("You answer questions about company financial reports. Use only the given text and "
          "table. Reply with the final answer only: a number (no units words) or a short phrase.")


def table_to_text(table: list[list[str]]) -> str:
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in table)


def direct_messages(record: Record) -> tuple[tuple[str, str], ...]:
    context = "\n".join(record.text)
    user = f"Text:\n{context}\n\nTable:\n{table_to_text(record.table)}\n\nQuestion: {record.question}"
    return (("system", SYSTEM), ("user", user))


STRUCTURED_SYSTEM = (
    "You answer questions about company financial reports. Use only the given text and table. "
    "Reply with one JSON object and nothing else, with these keys:\n"
    '  "values": the numbers you took from the report, as a list of {"name": short label, '
    '"value": number exactly as written in the report}. Empty list if none.\n'
    '  "formula": the calculation using those numbers, in plain arithmetic with + - * / and '
    'parentheses, e.g. "(5829 - 5735) / 5735 * 100". Empty string if no calculation.\n'
    '  "answer": the final answer, a number (no unit words) or a short phrase; a list of '
    "phrases if the question asks for several items."
)


def structured_messages(record: Record) -> tuple[tuple[str, str], ...]:
    user = direct_messages(record)[1][1]
    return (("system", STRUCTURED_SYSTEM), ("user", user))
