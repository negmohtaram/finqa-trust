"""Plot how the 95% interval for one accuracy narrows as the number of questions grows.

Usage: python scripts/plot_ci_width.py [--accuracy 0.6]   (needs the plots extra)
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from finqa_trust.stats import bootstrap_ci, ci_half_width  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "ci_width.png"
SIZES = [25, 50, 100, 200, 400, 800, 1600, 3200]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy", type=float, default=0.6)
    args = parser.parse_args()
    p = args.accuracy

    rng = random.Random(0)
    boot = []
    for n in SIZES:
        sample = [rng.random() < p for _ in range(n)]
        _, lo, hi = bootstrap_ci(sample, n_boot=1000)
        boot.append(100 * (hi - lo) / 2)
    dense = range(20, 3300, 10)
    formula = [100 * ci_half_width(p, n) for n in dense]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(list(dense), formula, color="#5b7cfa", label="formula: 1.96·√(p(1−p)/n)")
    ax.scatter(SIZES, boot, color="#9b5de5", zorder=3, label="bootstrap on a simulated sample")
    ax.axhline(3, color="#f4a261", linestyle="--", label="±3 points")
    for n in (100, 400):
        ax.annotate(f"n={n}: ±{ci_half_width(p, n) * 100:.1f}", (n, ci_half_width(p, n) * 100),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("questions (log scale)")
    ax.set_ylabel("95% interval half width (accuracy points)")
    ax.set_title(f"How precise is one accuracy number? (accuracy ≈ {p:.0%})")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    OUT.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
