"""Before any search runs: how much gold evidence can each chunking strategy return at all?"""
import statistics

from finqa_trust.chunks import STRATEGIES, chunk_page, holds_evidence, load_pages


def main() -> None:
    items = load_pages("dev")
    pages = {item["filename"]: item for item in items}
    print(f"{len(items)} questions, {len(pages)} pages\n")
    print(f"{'strategy':<12} {'chunks':>7} {'median':>7} {'max':>6} {'over 200':>9} {'gold held':>10} {'all gold':>9}")

    for strategy in STRATEGIES:
        by_page = {name: chunk_page(item, strategy) for name, item in pages.items()}
        sizes = [len(c.text.split()) for chunks in by_page.values() for c in chunks]
        held = total = complete = 0
        for item in items:
            gold = list(item["qa"]["gold_inds"])
            found = [any(holds_evidence(c, g) for c in by_page[item["filename"]]) for g in gold]
            held += sum(found)
            total += len(gold)
            complete += all(found)
        print(
            f"{strategy:<12} {len(sizes):>7} {statistics.median(sizes):>7.0f} {max(sizes):>6}"
            f" {sum(s > 200 for s in sizes):>9} {held / total:>10.1%} {complete / len(items):>9.1%}"
        )


if __name__ == "__main__":
    main()
