"""Measure what the content-hash cache saves when pages are edited or added.

Each chunking gets its own empty cache, so the numbers do not depend on what an earlier run left behind.
The real cache in results/embeddings is never read or written.
Usage: python scripts/measure_incremental.py [model]
"""
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from finqa_trust.chunks import STRATEGIES, chunk_page, load_pages
from finqa_trust.embed import embed_texts, text_key

MODEL = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/all-MiniLM-L6-v2"
ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results" / "incremental.json"
OUT_MD = ROOT / "results" / "incremental.md"
SEED = 0
EDITED_PAGES = 50
NEW_PAGES = 20
INSERT = "in that year"  # three words, the size of a small correction


def text_lines(item: dict) -> list[tuple[str, int]]:
    """(field, index) of every real sentence, in reading order; OCR full stops are skipped."""
    return [(field, i) for field in ("pre_text", "post_text")
            for i, line in enumerate(item[field]) if line.strip() != "."]


def edit_page(item: dict, where: str, rng: random.Random) -> dict:
    """Insert three words in the middle of one sentence: the first, the last or a random one."""
    lines = text_lines(item)
    field, i = {"top": lines[0], "bottom": lines[-1]}.get(where) or rng.choice(lines)
    edited = copy.deepcopy(item)
    words = edited[field][i].split()
    edited[field][i] = " ".join(words[:len(words) // 2] + INSERT.split() + words[len(words) // 2:])
    return edited


class CountingEncoder:
    def __init__(self, model: SentenceTransformer):
        self.model, self.count = model, 0

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.count += len(texts)
        return self.model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)


def index(pages: list[dict], strategy: str, encode: CountingEncoder, cache: dict) -> tuple[int, float, list[str]]:
    """Chunk and embed every page the way embed_chunks.py does. Returns (texts encoded, seconds, chunk texts)."""
    before = encode.count
    started = time.perf_counter()
    texts = [c.text for item in pages for c in chunk_page(item, strategy)]
    embed_texts(texts, encode, cache)
    return encode.count - before, time.perf_counter() - started, texts


def per_page_reembeds(pages: list[dict], edited: dict[str, dict], strategy: str, encode, cache: dict) -> list[int]:
    """Chunks re-embedded for each edited page, from a copy of the warm cache so pages do not share work."""
    counts = []
    for name, item in edited.items():
        trial = dict(cache)
        before = encode.count
        embed_texts([c.text for c in chunk_page(item, strategy)], encode, trial)
        counts.append(encode.count - before)
    return counts


def measure(strategy: str, pages: list[dict], encode: CountingEncoder) -> dict:
    rng = random.Random(SEED)
    held_out = rng.sample(range(len(pages)), NEW_PAGES)
    base = [p for i, p in enumerate(pages) if i not in set(held_out)]
    new = [pages[i] for i in held_out]

    cache: dict = {}
    cold_n, cold_s, _ = index(base, strategy, encode, cache)
    warm_n, warm_s, _ = index(base, strategy, encode, cache)

    editable = [i for i, p in enumerate(base) if text_lines(p)]
    chosen = rng.sample(editable, EDITED_PAGES)
    row = {"strategy": strategy, "pages": len(base), "chunks": cold_n,
           "cold": {"encoded": cold_n, "seconds": round(cold_s, 2)},
           "warm": {"encoded": warm_n, "seconds": round(warm_s, 2)}}

    for where in ("random", "top", "bottom"):
        edit_rng = random.Random(SEED)
        edited = {base[i]["filename"]: edit_page(base[i], where, edit_rng) for i in chosen}
        counts = per_page_reembeds(base, edited, strategy, encode, cache)
        sizes = [len(chunk_page(base[i], strategy)) for i in chosen]
        row[f"edit_{where}"] = {
            "mean": round(float(np.mean(counts)), 2),
            "max": int(max(counts)),
            "share_of_page": round(sum(counts) / sum(sizes), 3),
            "more_than_one": sum(c > 1 for c in counts),
        }

    # Whole corpus after the random edits: what a real reindex run pays, and what it leaves behind
    edit_rng = random.Random(SEED)
    after = [edit_page(p, "random", edit_rng) if i in set(chosen) else p for i, p in enumerate(base)]
    trial = dict(cache)
    reindex_n, reindex_s, live = index(after, strategy, encode, trial)
    live_keys = {text_key(t) for t in live}
    row["reindex_after_edits"] = {"encoded": reindex_n, "seconds": round(reindex_s, 2),
                                  "stale_keys": sum(k not in live_keys for k in trial)}

    add_n, add_s, _ = index(base + new, strategy, encode, cache)
    row["add_pages"] = {"pages": NEW_PAGES, "encoded": add_n, "seconds": round(add_s, 2),
                        "their_chunks": sum(len(chunk_page(p, strategy)) for p in new)}
    return row


def markdown(rows: list[dict]) -> str:
    lines = [f"# Incremental indexing ({MODEL.split('/')[-1]}, seed {SEED})", "",
             "| chunking | unique chunks | cold s | warm encoded | warm s | reindex after 50 edits: encoded | s | stale keys |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e = r["reindex_after_edits"]
        lines.append(f"| {r['strategy']} | {r['chunks']} | {r['cold']['seconds']} | {r['warm']['encoded']} "
                     f"| {r['warm']['seconds']} | {e['encoded']} | {e['seconds']} | {e['stale_keys']} |")
    lines += ["", f"Chunks re-embedded per edited page ({EDITED_PAGES} pages, \"{INSERT}\" inserted mid-sentence)", "",
              "| chunking | where | mean | max | share of the page | pages with more than one |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        for where in ("random", "top", "bottom"):
            e = r[f"edit_{where}"]
            lines.append(f"| {r['strategy']} | {where} | {e['mean']} | {e['max']} | {e['share_of_page']:.0%} "
                         f"| {e['more_than_one']} |")
    lines += ["", f"Adding {NEW_PAGES} new pages", "", "| chunking | encoded | their chunks | s |", "|---|---|---|---|"]
    for r in rows:
        a = r["add_pages"]
        lines.append(f"| {r['strategy']} | {a['encoded']} | {a['their_chunks']} | {a['seconds']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    encode = CountingEncoder(SentenceTransformer(MODEL))
    pages = list({item["filename"]: item for item in load_pages("dev")}.values())  # many questions share a page
    rows = []
    for strategy in STRATEGIES:
        rows.append(measure(strategy, pages, encode))
        print(json.dumps(rows[-1]))
    OUT_JSON.write_text(json.dumps(rows, indent=1) + "\n")
    OUT_MD.write_text(markdown(rows))
    print("\n" + markdown(rows))


if __name__ == "__main__":
    main()
