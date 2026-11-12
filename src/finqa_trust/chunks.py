"""Split report pages into chunks and check which gold evidence each chunk really holds."""
import json
from pathlib import Path

from pydantic import BaseModel

from .data import FILES, RAW

HEADER = "table_0"


class Unit(BaseModel):
    """One sentence or table row, with the id FinQA's gold_inds uses for it."""

    id: str
    text: str


class Chunk(BaseModel):
    page: str
    strategy: str
    order: int
    text: str
    units: list[str]  # only units held whole; a unit cut at the edge is not listed


def page_units(item: dict) -> list[Unit]:
    """Page in reading order: text before the table, the rows, text after it."""
    pre, post = item["pre_text"], item["post_text"]

    def sentences(lines: list[str], start: int) -> list[Unit]:
        return [
            Unit(id=f"text_{start + i}", text=line.strip())
            for i, line in enumerate(lines)
            if line.strip() != "."  # OCR leaves lines holding one full stop; gold_inds still counts them
        ]

    rows = [Unit(id=f"table_{i}", text=" | ".join(row)) for i, row in enumerate(item["table"])]
    return sentences(pre, 0) + rows + sentences(post, len(pre))


def fixed_size(units: list[Unit], size: int = 150, overlap: int = 30) -> list[tuple[str, list[str]]]:
    """Sliding word window that ignores the page's structure."""
    words = [(word, unit.id) for unit in units for word in unit.text.split()]
    totals: dict[str, int] = {}
    for _, unit_id in words:
        totals[unit_id] = totals.get(unit_id, 0) + 1

    pieces = []
    step = size - overlap
    for start in range(0, len(words), step):
        window = words[start:start + size]
        counts: dict[str, int] = {}
        for _, unit_id in window:
            counts[unit_id] = counts.get(unit_id, 0) + 1
        whole = [unit_id for unit_id, n in counts.items() if n == totals[unit_id]]
        pieces.append((" ".join(word for word, _ in window), whole))
        if start + size >= len(words):
            break
    return pieces


def _group(units: list[Unit], max_words: int) -> list[tuple[str, list[str]]]:
    """Fill a chunk unit by unit, starting a new one when the next unit would not fit."""
    pieces, current, count = [], [], 0
    for unit in units:
        n = len(unit.text.split())
        if current and count + n > max_words:
            pieces.append(("\n".join(u.text for u in current), [u.id for u in current]))
            current, count = [], 0
        current.append(unit)
        count += n
    if current:
        pieces.append(("\n".join(u.text for u in current), [u.id for u in current]))
    return pieces


def by_sentence(units: list[Unit], max_words: int = 150) -> list[tuple[str, list[str]]]:
    """Never cut a sentence or row, but a long table can still spread over chunks."""
    return _group(units, max_words)


def table_whole(units: list[Unit], max_words: int = 150) -> list[tuple[str, list[str]]]:
    """Group the prose as by_sentence does; the table is always one chunk of its own."""
    rows = [u for u in units if u.id.startswith("table_")]
    if not rows:
        return _group(units, max_words)
    first = units.index(rows[0])
    before, after = units[:first], units[first + len(rows):]
    table = ("\n".join(u.text for u in rows), [u.id for u in rows])
    return _group(before, max_words) + [table] + _group(after, max_words)


STRATEGIES = {"fixed": fixed_size, "sentence": by_sentence, "table_whole": table_whole}


def chunk_page(item: dict, strategy: str) -> list[Chunk]:
    pieces = STRATEGIES[strategy](page_units(item))
    return [
        Chunk(page=item["filename"], strategy=strategy, order=i, text=text, units=units)
        for i, (text, units) in enumerate(pieces)
    ]


def holds_evidence(chunk: Chunk, gold_id: str) -> bool:
    """A sentence must be whole. A table row also needs the header, or its numbers have no labels."""
    if gold_id not in chunk.units:
        return False
    return not gold_id.startswith("table_") or HEADER in chunk.units


def load_pages(split: str = "dev", raw_dir: Path = RAW) -> list[dict]:
    """FinQA questions, each with its full page. One page can carry several questions."""
    return json.loads((raw_dir / FILES[("finqa", split)]).read_text())
