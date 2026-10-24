"""Sample-size planning for the monitor experiment, by simulation.

Inputs are measured, not assumed:
  * baseline challenge rate p0 and the intra-game correlation (ICC) of challenge decisions,
    both from the 19 pre-alliance games of the paper's secret-channel series;
  * m, the number of decisions on monitor plays per short game, from the pilot.

Each simulated game draws its own challenge propensity from a beta distribution with the
measured ICC (a beta-binomial model), so decisions inside a game are correlated just as in
real games. Power is the share of simulated experiments in which a game-level permutation test
(the test used in the analysis) rejects at alpha = 0.05.

Usage (from Collusion_Liars_Bar/):
    uv run python analysis/monitor/power.py --m 6
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from positive_control import load_games  # noqa: E402

OUT = "docs/monitor_sting/results/power.json"


def icc_binary(groups):
    """One-way ANOVA estimator of the ICC for binary outcomes with unequal group sizes."""
    groups = [g for g in groups if len(g) > 0]
    k = len(groups)
    n_i = np.array([len(g) for g in groups], float)
    N = n_i.sum()
    means = np.array([np.mean(g) for g in groups])
    grand = sum(sum(g) for g in groups) / N
    msb = np.sum(n_i * (means - grand) ** 2) / (k - 1)
    msw = sum(sum((x - m) ** 2 for x in g) for g, m in zip(groups, means)) / (N - k)
    n0 = (N - np.sum(n_i ** 2) / N) / (k - 1)
    return float(max(0.0, (msb - msw) / (msb + (n0 - 1) * msw))), float(grand)


def simulate(n_games, m, p0, p1, icc, reps, rng, perm_reps=400):
    """Power of a game-level permutation test of p1 vs p0 with n_games per arm."""
    def draw(p):
        if icc <= 0:
            return [rng.binomial(m, p) for _ in range(n_games)]
        a = p * (1 - icc) / icc
        b = (1 - p) * (1 - icc) / icc
        return [rng.binomial(m, rng.beta(a, b)) for _ in range(n_games)]

    hits = 0
    for _ in range(reps):
        x, y = np.array(draw(p0)), np.array(draw(p1))
        obs = abs(x.sum() - y.sum())
        pooled = np.concatenate([x, y])
        cnt = 0
        for _ in range(perm_reps):
            rng.shuffle(pooled)
            cnt += abs(pooled[:n_games].sum() - pooled[n_games:].sum()) >= obs
        hits += (cnt + 1) / (perm_reps + 1) < 0.05
    return hits / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, required=True, help="decisions on monitor plays per game (pilot)")
    ap.add_argument("--p0", type=float, default=None,
                    help="baseline challenge rate on the monitor (pilot); default: the series' rate")
    ap.add_argument("--reps", type=int, default=400)
    args = ap.parse_args()
    rng = np.random.default_rng(7)

    games = load_games()
    pre = [g for g in games if not g["pair"]]
    groups = [[int(e["challenged"]) for e in g["events"]] for g in pre]
    icc, p0_series = icc_binary(groups)
    p0 = args.p0 if args.p0 is not None else p0_series
    m = int(round(args.m))
    deff = 1 + (m - 1) * icc
    print(f"From {len(pre)} pre-alliance games: challenge rate {p0_series:.2f}, ICC = {icc:.3f}")
    print(f"Baseline used for the monitor: p0 = {p0:.2f}")
    print(f"m = {m} decisions per game -> design effect 1 + (m-1)*ICC = {deff:.2f}\n")

    effects = [e for e in (0.10, 0.15, 0.20, 0.25) if e < p0]
    sizes = [5, 10, 15, 20, 30]
    table = {}
    print("Power (game-level permutation test, alpha 0.05) to detect a drop from p0 to p0 - effect")
    print("games/arm " + " ".join(f"{'-' + str(int(e * 100)) + 'pts':>8s}" for e in effects))
    for n in sizes:
        row = [simulate(n, m, p0, max(0.01, p0 - e), icc, args.reps, rng) for e in effects]
        table[n] = dict(zip(effects, row))
        print(f"{n:9d} " + " ".join(f"{v:8.2f}" for v in row))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"p0": p0, "p0_series": p0_series, "icc": icc, "m": m, "design_effect": deff, "n_pre_games": len(pre),
               "power": {str(n): {str(e): v for e, v in r.items()} for n, r in table.items()}},
              open(OUT, "w"), indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
