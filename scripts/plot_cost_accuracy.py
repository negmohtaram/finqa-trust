"""Plot accuracy against list price per 1,000 questions for each approach and mark the Pareto front.

An approach is on the front when no other approach is at least as accurate for less money.
Error bars are 95% bootstrap intervals. Also prints the cost per 1,000 correct answers with the
range implied by the accuracy interval.

Usage: python scripts/plot_cost_accuracy.py   (needs the plots extra)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from compare import FILES, PRICE_IN, PRICE_OUT, RESULTS, load  # noqa: E402
from finqa_trust.stats import bootstrap_ci  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "cost_accuracy.png"


def pareto(points: dict[str, tuple[float, float]]) -> list[str]:
    """Names not beaten by another point that costs no more and is at least as accurate."""
    front = []
    for name, (cost, acc) in points.items():
        beaten = any(c <= cost and a >= acc and (c, a) != (cost, acc)
                     for other, (c, a) in points.items() if other != name)
        if not beaten:
            front.append(name)
    return sorted(front, key=lambda n: points[n][0])


def main() -> None:
    runs = {a: load(f) for a, f in FILES.items() if (RESULTS / f).exists()}
    points, bars = {}, {}
    print("approach | accuracy [95% CI] | $ per 1,000 questions | $ per 1,000 correct [range]")
    for name, rows in runs.items():
        n = len(rows)
        acc, lo, hi = bootstrap_ci([r["correct"] for r in rows])
        cost = sum(r["input_tokens"] * PRICE_IN + r["output_tokens"] * PRICE_OUT for r in rows) / n * 1000
        points[name] = (cost, acc)
        bars[name] = (acc - lo, hi - acc)
        print(f"{name} | {acc:.0%} [{lo:.0%}, {hi:.0%}] | ${cost:.2f} | "
              f"${cost / acc:.2f} [${cost / hi:.2f}, ${cost / lo:.2f}]")

    front = pareto(points)
    print("Pareto front:", ", ".join(front))

    # Approaches with identical results share one marker and one label ("alone = code").
    shared: dict[tuple[float, float], list[str]] = {}
    for name, point in points.items():
        shared.setdefault(point, []).append(name)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for names in shared.values():
        name = names[0]
        cost, acc = points[name]
        on = name in front
        ax.errorbar(cost, 100 * acc, yerr=[[100 * bars[name][0]], [100 * bars[name][1]]],
                    fmt="o" if on else "s", color="#3b6fd8" if on else "#999999", capsize=4)
        ax.annotate(" = ".join(names), (cost, 100 * acc), textcoords="offset points",
                    xytext=(8, 4) if on else (8, -14))
    ax.plot([points[n][0] for n in front], [100 * points[n][1] for n in front], "--", color="#3b6fd8")
    ax.set_xlabel("list price per 1,000 questions (USD)")
    ax.set_ylabel("correct (%)")
    ax.set_title(f"Accuracy vs cost on the same {len(runs['alone'])} questions")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
