"""Measure recall@10 against latency for pgvector's HNSW and IVF indexes on the chunk vectors.

Recall here is the overlap with the exact top 10 from numpy, not evidence recall.
Needs a Postgres with pgvector, for example:
  docker run -d --name finqa-vec -e POSTGRES_PASSWORD=finqa -e POSTGRES_DB=finqa -p 5434:5432 pgvector/pgvector:pg16
Usage: FINQA_PG_URL=postgresql://postgres:finqa@localhost:5434/finqa python scripts/index_tuning.py   (needs the index and plots extras)
"""
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import psycopg  # noqa: E402

from eval_retrieval import EMBEDDINGS, MINILM, question_vectors  # noqa: E402
from finqa_trust.chunks import STRATEGIES, load_pages  # noqa: E402

PG_URL = os.environ.get("FINQA_PG_URL", "postgresql://postgres:finqa@localhost:5434/finqa")
ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results" / "index_tuning.json"
OUT_PNG = ROOT / "docs" / "index_tuning.png"
K = 10
PASSES = 3
HNSW_M = (8, 16, 32)
HNSW_EF = (10, 20, 40, 80, 160, 400)
IVF_LISTS = 67  # about the square root of the number of vectors
IVF_PROBES = (1, 2, 4, 8, 16, 32, 67)


def to_pg(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


def load_table(conn: psycopg.Connection, vectors: np.ndarray) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(f"CREATE TABLE chunks (id int PRIMARY KEY, embedding vector({vectors.shape[1]}))")
    with conn.cursor().copy("COPY chunks (id, embedding) FROM STDIN") as copy:
        for i, vector in enumerate(vectors):
            copy.write_row((i, to_pg(vector)))
    conn.execute("ANALYZE chunks")


def run_queries(conn: psycopg.Connection, queries: list[str], truth: np.ndarray) -> dict:
    sql = "SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s"
    for q in queries[:50]:  # warm the cache so the first queries are not slower
        conn.execute(sql, (q, K)).fetchall()
    recalls, millis = [], []
    for q, exact in zip(queries, truth):
        times = []
        for _ in range(PASSES):  # the fastest pass is the least disturbed by other work on the laptop
            start = time.perf_counter()
            found = [row[0] for row in conn.execute(sql, (q, K)).fetchall()]
            times.append(1000 * (time.perf_counter() - start))
        millis.append(min(times))
        recalls.append(len(set(found) & set(exact.tolist())) / K)
    return {
        "recall": float(np.mean(recalls)),
        "ms_median": float(np.median(millis)),
        "ms_p95": float(np.percentile(millis, 95)),
    }


def plot(rows: list[dict], exact_ms: float) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {8: "#9b5de5", 16: "#5b7cfa", 32: "#00b4d8"}
    for m in HNSW_M:
        line = [r for r in rows if r["index"] == "hnsw" and r["m"] == m]
        ax.plot([r["ms_median"] for r in line], [r["recall"] for r in line], "o-", color=colors[m],
                label=f"HNSW m={m} (ef_search {HNSW_EF[0]}→{HNSW_EF[-1]})")
    line = [r for r in rows if r["index"] == "ivf"]
    ax.plot([r["ms_median"] for r in line], [r["recall"] for r in line], "s--", color="#f4a261",
            label=f"IVF lists={IVF_LISTS} (probes {IVF_PROBES[0]}→{IVF_PROBES[-1]})")
    for r in rows:
        if r["index"] == "hnsw" and r["m"] == 16:
            ax.annotate(f"ef {r['ef_search']}", (r["ms_median"], r["recall"]),
                        textcoords="offset points", xytext=(5, -12), fontsize=8)
    ax.axvline(exact_ms, color="#e76f51", linestyle=":", label=f"exact scan in Postgres ({exact_ms:.2f} ms)")
    ax.set_xlabel("median latency per question (ms)")
    ax.set_ylabel("recall@10 against exact search")
    ax.set_title("Recall against latency, 4,552 MiniLM vectors, 883 questions")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    OUT_PNG.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


def main() -> None:
    folder = EMBEDDINGS / MINILM.split("/")[-1]
    vectors = np.concatenate([np.load(folder / f"{s}.npy") for s in STRATEGIES]).astype(np.float32)
    # The same text appears in several chunk sets. Identical vectors tie, and an index may return
    # another copy than numpy did, which would count as a miss. Keep one copy of each.
    vectors = np.unique(vectors, axis=0)
    questions = [item["qa"]["question"] for item in load_pages("dev")]
    qvecs = question_vectors(MINILM, questions).astype(np.float32)
    truth = np.argsort(-(qvecs @ vectors.T), axis=1, kind="stable")[:, :K]
    queries = [to_pg(q) for q in qvecs]
    print(f"{len(vectors)} vectors, {len(queries)} questions")

    rows = []
    with psycopg.connect(PG_URL, autocommit=True) as conn:
        load_table(conn, vectors)
        exact = run_queries(conn, queries, truth)
        print(f"exact scan: recall {exact['recall']:.3f}, {exact['ms_median']:.2f} ms")
        conn.execute("SET enable_seqscan = off")

        for m in HNSW_M:
            start = time.perf_counter()
            conn.execute(f"CREATE INDEX chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) "
                         f"WITH (m = {m}, ef_construction = 64)")
            build_s = time.perf_counter() - start
            for ef in HNSW_EF:
                conn.execute(f"SET hnsw.ef_search = {ef}")
                row = {"index": "hnsw", "m": m, "ef_search": ef, "build_s": build_s,
                       **run_queries(conn, queries, truth)}
                rows.append(row)
                print(row)
            conn.execute("DROP INDEX chunks_hnsw")

        start = time.perf_counter()
        conn.execute(f"CREATE INDEX chunks_ivf ON chunks USING ivfflat (embedding vector_cosine_ops) "
                     f"WITH (lists = {IVF_LISTS})")
        build_s = time.perf_counter() - start
        for probes in IVF_PROBES:
            conn.execute(f"SET ivfflat.probes = {probes}")
            row = {"index": "ivf", "lists": IVF_LISTS, "probes": probes, "build_s": build_s,
                   **run_queries(conn, queries, truth)}
            rows.append(row)
            print(row)
        conn.execute("DROP INDEX chunks_ivf")

    OUT_JSON.write_text(json.dumps({"exact": exact, "rows": rows}, indent=1))
    plot(rows, exact["ms_median"])


if __name__ == "__main__":
    main()
