"""Turn chunk texts into unit vectors, reusing any text already embedded."""
import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Unit length, so a dot product between two vectors is their cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def embed_texts(
    texts: list[str],
    encode: Callable[[list[str]], np.ndarray],
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    """One row per text, in input order. Only texts missing from the cache reach encode."""
    keys = [text_key(text) for text in texts]
    missing = {key: text for key, text in zip(keys, texts) if key not in cache}
    if missing:
        fresh = normalize(encode(list(missing.values())))
        cache.update(zip(missing, fresh))
    return np.stack([cache[key] for key in keys])


def load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path) as saved:
        return dict(zip(saved["keys"].tolist(), saved["vectors"]))


def save_cache(cache: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, keys=np.array(list(cache)), vectors=np.stack(list(cache.values())))
