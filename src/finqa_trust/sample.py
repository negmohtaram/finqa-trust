"""Fixed, stratified samples of the dev set, so every approach is compared on the same questions."""
import json
import random
from collections import defaultdict
from pathlib import Path

from .schema import Record

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


def stratified_sample(records: list[Record], n: int, seed: int = 0) -> list[Record]:
    """Pick n records with the same mix of (source, answer type) as the full list.

    Each group gets its proportional share rounded down; the seats left over go to the
    groups with the largest remainders. Order inside the result is shuffled with the seed.
    """
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in records:
        groups[(r.source, r.answer_type)].append(r)
    keys = sorted(groups)
    exact = {k: n * len(groups[k]) / len(records) for k in keys}
    counts = {k: int(exact[k]) for k in keys}
    by_remainder = sorted(keys, key=lambda k: exact[k] - counts[k], reverse=True)
    for k in by_remainder[: n - sum(counts.values())]:
        counts[k] += 1
    rng = random.Random(seed)
    picked = [r for k in keys for r in rng.sample(groups[k], counts[k])]
    rng.shuffle(picked)
    return picked


def save_ids(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.id for r in records], indent=1) + "\n")


def load_ids(path: Path, records: list[Record]) -> list[Record]:
    """Rebuild a saved sample, in its saved order. A missing id is an error, not a skip."""
    by_id = {r.id: r for r in records}
    ids = json.loads(path.read_text())
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} sampled ids not in the data, e.g. {missing[0]}")
    return [by_id[i] for i in ids]
