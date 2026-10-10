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
