"""Small statistics helpers shared by the monitor-experiment analyses.

Challenge decisions inside one game are not independent (same hands, same opinions, same
history), so every interval and test here resamples or permutes whole games, never single
decisions.
"""
import math
import random
from typing import Callable, Dict, List, Sequence, Tuple

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> Tuple[float, float, float]:
    """Proportion with a Wilson score interval (behaves at 0/n and n/n, unlike the Wald one)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def rate(events: Sequence[Dict], key: str = "challenged") -> Tuple[int, int]:
    return sum(bool(e[key]) for e in events), len(events)


def cluster_bootstrap(groups: List[List[Dict]], stat: Callable[[List[Dict]], float],
                      reps: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    """Point estimate and percentile 95% CI, resampling games (each group = one game's events)."""
    rng = random.Random(seed)
    flat = [e for g in groups for e in g]
    point = stat(flat)
    boots = []
    for _ in range(reps):
        sample = [e for g in (rng.choice(groups) for _ in groups) for e in g]
        v = stat(sample)
        if not math.isnan(v):
            boots.append(v)
    boots.sort()
    if not boots:
        return point, float("nan"), float("nan")
    return point, boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]


def permutation_test(a: List[List[Dict]], b: List[List[Dict]], stat: Callable[[List[Dict]], float],
                     reps: int = 10000, seed: int = 0) -> Tuple[float, float]:
    """Two-sided test of stat(a) - stat(b) = 0, permuting game labels between the two conditions."""
    rng = random.Random(seed)
    obs = stat([e for g in a for e in g]) - stat([e for g in b for e in g])
    pooled = a + b
    hits = 0
    valid = 0
    for _ in range(reps):
        rng.shuffle(pooled)
        pa, pb = pooled[: len(a)], pooled[len(a):]
        d = stat([e for g in pa for e in g]) - stat([e for g in pb for e in g])
        if math.isnan(d):
            continue
        valid += 1
        hits += abs(d) >= abs(obs) - 1e-12
    return obs, (hits + 1) / (valid + 1)


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value for a 2x2 table (k1/n1 vs k2/n2)."""
    from scipy.stats import fisher_exact as fe
    return float(fe([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control for the confirmatory tests)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def fmt_ci(p: float, lo: float, hi: float, pct: bool = True) -> str:
    if math.isnan(p):
        return "n/a"
    if pct:
        return f"{100 * p:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"
    return f"{p:+.2f} [{lo:+.2f}, {hi:+.2f}]"
