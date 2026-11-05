"""Find the chunks that hold a question's evidence: keyword search, vector search and their fusion."""
import math
import re
from collections import Counter, defaultdict

import numpy as np

from .chunks import Chunk, holds_evidence

TOKEN = re.compile(r"[a-z]+|\d[\d,.]*\d|\d")  # keeps 1,234.5 as one token so a figure can match exactly


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    """Keyword score: rare query words count most, repeats saturate, long chunks pay a penalty."""

    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        docs = [Counter(tokenize(text)) for text in texts]
        self.lengths = np.array([sum(doc.values()) for doc in docs], dtype=float)
        self.average = self.lengths.mean() if len(docs) else 0.0
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, count in doc.items():
                self.postings[term].append((i, count))
        self.size = len(docs)

    def scores(self, query: str) -> np.ndarray:
        """One score per chunk; 0 means no query word appears in it."""
        out = np.zeros(self.size)
        for term in set(tokenize(query)):
            hits = self.postings.get(term)
            if not hits:
                continue
            idf = math.log(1 + (self.size - len(hits) + 0.5) / (len(hits) + 0.5))
            ids = np.array([i for i, _ in hits])
            counts = np.array([count for _, count in hits], dtype=float)
            penalty = 1 - self.b + self.b * self.lengths[ids] / self.average
            out[ids] += idf * counts * (self.k1 + 1) / (counts + self.k1 * penalty)
        return out


def rank(scores: np.ndarray, allowed: np.ndarray, depth: int = 50) -> list[int]:
    """Chunk positions, best first, among the allowed ones. Ties keep chunk order."""
    order = np.argsort(-scores, kind="stable")
    return [int(i) for i in order[allowed[order]][:depth]]


def fuse(rankings: list[list[int]], k: int = 60, depth: int = 50) -> list[int]:
    """Reciprocal rank fusion. Uses positions only, because the methods' scores share no scale."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for position, chunk in enumerate(ranking, start=1):
            scores[chunk] += 1 / (k + position)
    return sorted(scores, key=lambda chunk: (-scores[chunk], chunk))[:depth]


def recall_at_k(ranking: list[int], chunks: list[Chunk], gold: list[str], page: str, k: int) -> float:
    """Share of the gold units held by some chunk in the top k. Unit ids repeat across pages, so the page must match."""
    top = [chunks[i] for i in ranking[:k] if chunks[i].page == page]
    return sum(any(holds_evidence(chunk, g) for chunk in top) for g in gold) / len(gold)


def reciprocal_rank(ranking: list[int], chunks: list[Chunk], gold: list[str], page: str, cutoff: int = 10) -> float:
    """1 / position of the first chunk holding any gold unit; 0 if none in the cutoff."""
    for position, i in enumerate(ranking[:cutoff], start=1):
        if chunks[i].page == page and any(holds_evidence(chunks[i], g) for g in gold):
            return 1 / position
    return 0.0
