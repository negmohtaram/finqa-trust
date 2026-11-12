"""Embed every chunk set and report how much text the model never reads."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from finqa_trust.chunks import STRATEGIES, chunk_page, load_pages
from finqa_trust.embed import embed_texts, load_cache, save_cache

MODEL = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/all-MiniLM-L6-v2"
OUT = Path(__file__).resolve().parents[1] / "results" / "embeddings" / MODEL.split("/")[-1]


def main() -> None:
    model = SentenceTransformer(MODEL)
    limit = model.max_seq_length
    pages = {item["filename"]: item for item in load_pages("dev")}
    cache_path = OUT / "cache.npz"
    cache = load_cache(cache_path)
    encoded = 0

    def encode(texts: list[str]) -> np.ndarray:
        nonlocal encoded
        encoded += len(texts)
        return model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)

    print(f"model reads at most {limit} tokens per chunk\n")
    print(f"{'strategy':<12} {'chunks':>7} {'cut off':>8} {'tokens lost':>11} {'seconds':>8}")
    OUT.mkdir(parents=True, exist_ok=True)
    for strategy in STRATEGIES:
        chunks = [c for item in pages.values() for c in chunk_page(item, strategy)]
        texts = [c.text for c in chunks]

        started = time.perf_counter()
        vectors = embed_texts(texts, encode, cache)
        seconds = time.perf_counter() - started

        ids = model.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
        tokens = np.array([len(i) for i in ids])
        cut = tokens > limit
        lost = np.clip(tokens - limit, 0, None).sum() / tokens.sum()

        np.save(OUT / f"{strategy}.npy", vectors)
        with (OUT / f"{strategy}.jsonl").open("w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.model_dump()) + "\n")
        print(f"{strategy:<12} {len(chunks):>7} {cut.sum():>8} {lost:>11.1%} {seconds:>8.1f}")

    save_cache(cache, cache_path)
    print(f"\nencoded {encoded} texts this run, cache holds {len(cache)}")


if __name__ == "__main__":
    main()
