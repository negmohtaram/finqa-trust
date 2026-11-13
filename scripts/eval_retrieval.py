"""Compare keyword, vector and fused search on every chunk set, on the same FinQA dev questions."""
import json
import random
from pathlib import Path

import numpy as np

from finqa_trust.chunks import STRATEGIES, Chunk, load_pages
from finqa_trust.embed import embed_texts, load_cache, save_cache
from finqa_trust.retrieval import BM25, fuse, rank, recall_at_k, reciprocal_rank
from finqa_trust.stats import bootstrap_ci, paired_bootstrap_diff

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MPNET = "sentence-transformers/all-mpnet-base-v2"
RESULTS = Path(__file__).resolve().parents[1] / "results"
EMBEDDINGS = RESULTS / "embeddings"
KS = (1, 3, 5, 10)


def company(page: str) -> str:
    return page.split("/")[0]


def load_chunks(strategy: str) -> list[Chunk]:
    path = EMBEDDINGS / MINILM.split("/")[-1] / f"{strategy}.jsonl"
    return [Chunk.model_validate_json(line) for line in path.read_text().splitlines()]


def question_vectors(model_name: str, questions: list[str]) -> np.ndarray:
    folder = EMBEDDINGS / model_name.split("/")[-1]
    cache = load_cache(folder / "cache.npz")

    def encode(texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        return model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)

    vectors = embed_texts(questions, encode, cache)
    save_cache(cache, folder / "cache.npz")
    return vectors


def score_run(run: list[list[int]], chunks: list[Chunk], items: list[dict]) -> dict[str, list[float]]:
    """Per-question metrics, in question order, so any two runs can be compared pairwise."""
    out: dict[str, list[float]] = {f"recall@{k}": [] for k in KS}
    out["complete@5"], out["mrr"], out["words@5"] = [], [], []
    for ranking, item in zip(run, items):
        gold, page = list(item["qa"]["gold_inds"]), item["filename"]
        for k in KS:
            out[f"recall@{k}"].append(recall_at_k(ranking, chunks, gold, page, k))
        out["complete@5"].append(float(out["recall@5"][-1] == 1))
        out["mrr"].append(reciprocal_rank(ranking, chunks, gold, page))
        out["words@5"].append(sum(len(chunks[i].text.split()) for i in ranking[:5]))
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def diff_cell(a: list[float], b: list[float]) -> str:
    diff, lo, hi = paired_bootstrap_diff(a, b)
    mark = " *" if lo > 0 or hi < 0 else ""
    return f"{diff * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]{mark}"


def main_table(scores: dict, scope: str, methods: list[str]) -> list[str]:
    lines = [
        "| chunking | method | recall@1 | recall@3 | recall@5 (95% interval) | recall@10 | complete@5 | MRR "
        "| words in top 5 | recall@5 vs dense | MRR vs dense |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for strategy in STRATEGIES:
        dense = scores[scope, strategy, "dense"]
        for method in methods:
            row = scores.get((scope, strategy, method))
            if row is None:
                continue
            _, lo, hi = bootstrap_ci(row["recall@5"])
            same = method == "dense"
            lines.append(
                f"| {strategy} | {method} | {mean(row['recall@1']):.3f} | {mean(row['recall@3']):.3f} "
                f"| {mean(row['recall@5']):.3f} [{lo:.3f}, {hi:.3f}] | {mean(row['recall@10']):.3f} "
                f"| {mean(row['complete@5']):.3f} | {mean(row['mrr']):.3f} | {mean(row['words@5']):.0f} "
                f"| {'' if same else diff_cell(dense['recall@5'], row['recall@5'])} "
                f"| {'' if same else diff_cell(dense['mrr'], row['mrr'])} |"
            )
    return lines


def evidence_kind(item: dict) -> str:
    kinds = {gold.split("_")[0] for gold in item["qa"]["gold_inds"]}
    return "both" if len(kinds) == 2 else kinds.pop()


def wins(row_scores: dict, items: list[dict]) -> list[str]:
    """Questions where one method finds strictly more evidence in the top 5 than the other two."""
    lines = []
    methods = list(row_scores)
    for method in methods:
        others = [m for m in methods if m != method]
        won = [
            i for i in range(len(items))
            if all(row_scores[method]["recall@5"][i] > row_scores[o]["recall@5"][i] for o in others)
        ]
        lines.append(f"\n**{method} alone finds more on {len(won)} questions.**")
        if won:
            i = max(won, key=lambda j: row_scores[method]["recall@5"][j])
            cells = ", ".join(f"{m} {row_scores[m]['recall@5'][i]:.2f}" for m in methods)
            lines.append(f'Example: "{items[i]["qa"]["question"]}" ({items[i]["id"]}). Recall@5: {cells}.')
    return lines


def write_misses(run: list[list[int]], chunks: list[Chunk], items: list[dict], scores: dict) -> None:
    pages = {(c.page, unit): c for c in chunks for unit in c.units}
    misses = []
    for i, item in enumerate(items):
        if scores["recall@5"][i] == 1:
            continue
        top = [chunks[j] for j in run[i][:5]]
        gold = item["qa"]["gold_inds"]
        missing = [g for g in gold if recall_at_k(list(range(5)), top, [g], item["filename"], 5) == 0]
        misses.append({
            "id": item["id"], "question": item["qa"]["question"], "missing": {g: gold[g] for g in missing},
            "found": [g for g in gold if g not in missing], "top_pages": [c.page for c in top],
            "top_chunks": [c.text[:300] for c in top[:3]],
            "reachable": [g for g in missing if (item["filename"], g) in pages],
        })
    (RESULTS / "retrieval_misses.jsonl").write_text("".join(json.dumps(m) + "\n" for m in misses))

    sample = random.Random(0).sample(misses, min(10, len(misses)))
    lines = [f"# 10 of {len(misses)} misses (table_whole, fusion, inside the company)\n"]
    for n, miss in enumerate(sample, start=1):
        lines.append(f"## {n}. {miss['question']}\n`{miss['id']}` · found: {miss['found'] or 'nothing'}\n")
        lines.append("Missing evidence:")
        lines += [f"- `{g}`: {text}" for g, text in miss["missing"].items()]
        lines.append(f"\nPages in the top 5: {', '.join(miss['top_pages'])}\n\nTop 3 chunks returned:")
        lines += [f"- {text!r}" for text in miss["top_chunks"]]
        lines.append("")
    (RESULTS / "retrieval_misses.md").write_text("\n".join(lines))


def main() -> None:
    items = load_pages("dev")
    questions = [item["qa"]["question"] for item in items]
    companies = [company(item["filename"]) for item in items]
    models = {"dense": MINILM}
    if (EMBEDDINGS / MPNET.split("/")[-1] / "table_whole.npy").exists():
        models["dense_mpnet"] = MPNET
    query_vectors = {name: question_vectors(model, questions) for name, model in models.items()}

    scores, runs, every_chunks = {}, {}, {}
    rankings_path = RESULTS / "retrieval_rankings.jsonl"
    with rankings_path.open("w") as saved:
        for strategy in STRATEGIES:
            chunks = load_chunks(strategy)
            every_chunks[strategy] = chunks
            bm25 = BM25([c.text for c in chunks])
            owners = np.array([company(c.page) for c in chunks])
            vectors = {
                name: np.load(EMBEDDINGS / model.split("/")[-1] / f"{strategy}.npy")
                for name, model in models.items()
            }
            for scope in ("company", "global"):
                current: dict[str, list[list[int]]] = {"bm25": []}
                for name in models:
                    current[name] = []
                    current[name.replace("dense", "fusion")] = []
                for q, question in enumerate(questions):
                    allowed = owners == companies[q] if scope == "company" else np.ones(len(chunks), dtype=bool)
                    keyword = bm25.scores(question)
                    current["bm25"].append(rank(keyword, allowed & (keyword > 0)))
                    for name in models:
                        dense = rank(vectors[name] @ query_vectors[name][q], allowed)
                        current[name].append(dense)
                        current[name.replace("dense", "fusion")].append(fuse([current["bm25"][q], dense]))
                for method, run in current.items():
                    scores[scope, strategy, method] = score_run(run, chunks, items)
                    runs[scope, strategy, method] = run
                    for item, ranking in zip(items, run):
                        saved.write(json.dumps({
                            "id": item["id"], "scope": scope, "strategy": strategy,
                            "method": method, "ranking": ranking,
                        }) + "\n")

    methods = ["bm25", "dense", "fusion", "dense_mpnet", "fusion_mpnet"]
    out = [
        "# Retrieval comparison", "",
        f"{len(items)} FinQA dev questions, the same ones in every row, with evidence from `gold_inds`. "
        "A table row counts only when its header is in the same chunk. Intervals are bootstrap over "
        "questions, 95%; differences are paired on the same questions, in points, and * means the "
        "interval excludes 0.", "",
        "- **recall@k**: share of a question's evidence in the top k chunks.",
        "- **complete@5**: all of it in the top 5.",
        "- **MRR**: 1 / rank of the first chunk with evidence, top 10 only.",
        "- **fusion**: reciprocal rank fusion of bm25 and dense, k = 60.", "",
        "## Search inside the question's company", "",
    ]
    out += main_table(scores, "company", methods)
    out += ["", "## Search over all companies", ""]
    out += main_table(scores, "global", methods)

    company_runs = {m: scores["company", "table_whole", m] for m in ("bm25", "dense", "fusion")}
    out += ["", "## Chunking and model, dense, inside the company", ""]
    fixed = scores["company", "fixed", "dense"]["recall@5"]
    whole = scores["company", "table_whole", "dense"]["recall@5"]
    sentence = scores["company", "sentence", "dense"]["recall@5"]
    out.append(f"- table_whole minus fixed, recall@5: {diff_cell(fixed, whole)}")
    out.append(f"- table_whole minus sentence, recall@5: {diff_cell(sentence, whole)}")
    if "dense_mpnet" in models:
        mpnet = scores["company", "table_whole", "dense_mpnet"]
        out.append(f"- mpnet minus MiniLM on table_whole, recall@5: {diff_cell(whole, mpnet['recall@5'])}")
        out.append(f"- mpnet minus MiniLM on table_whole, MRR: "
                   f"{diff_cell(company_runs['dense']['mrr'], mpnet['mrr'])}")

    out += ["", "## By where the evidence sits (table_whole, inside the company, recall@5)", ""]
    out += ["| evidence | questions | bm25 | dense | fusion |", "|---|---|---|---|---|"]
    for kind in ("table", "text", "both"):
        index = [i for i, item in enumerate(items) if evidence_kind(item) == kind]
        cells = " | ".join(
            f"{mean([company_runs[m]['recall@5'][i] for i in index]):.3f}" for m in ("bm25", "dense", "fusion")
        )
        out.append(f"| {kind} | {len(index)} | {cells} |")

    out += ["", "## Where each method wins (table_whole, inside the company)"]
    out += wins(company_runs, items)
    (RESULTS / "retrieval.md").write_text("\n".join(out) + "\n")
    write_misses(runs["company", "table_whole", "fusion"], every_chunks["table_whole"], items, company_runs["fusion"])
    print("\n".join(out))


if __name__ == "__main__":
    main()
