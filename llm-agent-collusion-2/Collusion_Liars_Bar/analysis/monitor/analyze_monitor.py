"""Analysis of the monitor-pact experiment (src/game/game_monitor.py).

Reads experiments/monitor_records/<style>/*.json and writes:
  docs/monitor_sting/results/summary.json   every number reported
  docs/monitor_sting/results/report.txt     the same, readable
  docs/monitor_sting/figures/*.png          the figures

Confirmatory tests (pre-registered in docs/monitor_sting/PREREGISTRATION.md):
  H1  verbal commitment rises with explicitness: benign < hint < direct (Fisher exact, one per
      adjacent pair).
  H2  agents challenge the monitor less after an offer than with no offer (game-level
      permutation test on the challenge rate for monitor plays, each offer style vs none).
  Holm correction across the five confirmatory tests.
Everything else is exploratory and labelled as such.

Usage (from Collusion_Liars_Bar/):
    uv run python analysis/monitor/analyze_monitor.py
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from stats_utils import (cluster_bootstrap, fisher_exact, fmt_ci, holm,  # noqa: E402
                         permutation_test, wilson)

STYLES = ["none", "benign", "hint", "direct"]
LABEL = {"none": "No offer", "benign": "Benign request", "hint": "Indirect hint", "direct": "Direct pact"}
MODELS = {"Lily": "Llama-3.1-8B", "Mike": "Ministral-8B", "Quinn": "Qwen2.5-7B"}
COLOR = {"none": "#7a7f85", "benign": "#4f8fbf", "hint": "#d9a43b", "direct": "#c4473a"}
RES = "docs/monitor_sting/results"
FIG = "docs/monitor_sting/figures"
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def load():
    games = defaultdict(list)
    for style in STYLES:
        for f in sorted(glob.glob(f"experiments/monitor_records/{style}/*.json")):
            d = json.load(open(f))
            if d["config"].get("pilot"):
                continue
            games[style].append(d)
    return games


def ch_rate(evs):
    return sum(e["challenged"] for e in evs) / len(evs) if evs else float("nan")


def decisions(game, on_monitor=True, bluff=None, challenger=None, include_fallback=False):
    out = []
    for e in game["challenge_events"]:
        if not e["challenger_is_agent"] or e["pred_is_monitor"] != on_monitor:
            continue
        if bluff is not None and e["bluff"] != bluff:
            continue
        if challenger and e["challenger"] != challenger:
            continue
        if e["fallback"] and not include_fallback:
            continue
        out.append(dict(e, _game=game["game_id"]))
    return out


def offer_rows(games):
    rows = []
    for style, gs in games.items():
        for g in gs:
            for agent, o in g["offers"].items():
                j = o.get("judge") or {}
                rows.append({"style": style, "agent": agent, "game": g["game_id"],
                             "accept_word": o["self_decision"] == "accept",
                             "commits": bool(j.get("commits_no_challenge")),
                             "suspects": bool(j.get("suspects_test")),
                             "distrusts": bool(j.get("distrusts_sender")),
                             "fairness": bool(j.get("fairness_objection")),
                             "safety": bool(j.get("safety_refusal")),
                             "judged": bool(j)})
    return rows


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    games = load()
    summary = {"n_games": {s: len(games[s]) for s in STYLES}}
    say("MONITOR-PACT EXPERIMENT: RESULTS")
    say("=" * 64)
    say("Games per condition: " + ", ".join(f"{LABEL[s]} {len(games[s])}" for s in STYLES))
    ends = defaultdict(int)
    for s in STYLES:
        for g in games[s]:
            ends[(g["end_reason"] or "?").split(":")[0]] += 1
    say("End reasons: " + ", ".join(f"{k} {v}" for k, v in ends.items()))
    all_ev = [e for s in STYLES for g in games[s] for e in g["challenge_events"] if e["challenger_is_agent"]]
    fb = sum(e["fallback"] for e in all_ev)
    say(f"Agent challenge decisions: {len(all_ev)}; fallbacks (no valid model reply) {fb} "
        f"({100 * fb / max(1, len(all_ev)):.1f}%), excluded below")
    summary["fallback_share"] = fb / max(1, len(all_ev))

    # ------------------------------------------------------------------ 1. verbal
    rows = offer_rows(games)
    say("\n1. VERBAL RESPONSE TO THE OFFER (one row per agent per game)")
    say(f"{'condition':16s} {'n':>3s}  {'says ACCEPT':>16s}  {'commits to pact':>16s}  {'suspects test':>16s}")
    verbal = {}
    for s in ["benign", "hint", "direct"]:
        r = [x for x in rows if x["style"] == s]
        n = len(r)
        cells = {k: (sum(x[k] for x in r), n) for k in ["accept_word", "commits", "suspects", "distrusts",
                                                        "fairness", "safety"]}
        verbal[s] = {k: {"k": v[0], "n": v[1], "ci": wilson(*v)} for k, v in cells.items()}
        say(f"{LABEL[s]:16s} {n:3d}  {fmt_ci(*wilson(*cells['accept_word'])):>16s}  "
            f"{fmt_ci(*wilson(*cells['commits'])):>16s}  {fmt_ci(*wilson(*cells['suspects'])):>16s}")
    summary["verbal"] = verbal
    say("   by model (commits to pact; hint and direct pooled):")
    by_model = {}
    for a, m in MODELS.items():
        r = [x for x in rows if x["agent"] == a and x["style"] in ("hint", "direct")]
        k = sum(x["commits"] for x in r)
        by_model[a] = {"k": k, "n": len(r), "ci": wilson(k, len(r))}
        say(f"   {m:14s} {fmt_ci(*wilson(k, len(r)))} (n={len(r)})")
    summary["verbal_by_model"] = by_model

    # ------------------------------------------------------------------ 2. behaviour
    say("\n2. CHALLENGE DECISIONS ON THE MONITOR'S PLAYS (game-clustered 95% CIs)")
    say(f"{'condition':16s} {'n':>4s} {'challenge monitor':>20s} {'on monitor bluffs':>20s} "
        f"{'challenge agents':>20s} {'preference index':>22s}")
    behav = {}
    groups = {}
    for s in STYLES:
        gm = [decisions(g, True) for g in games[s]]
        gb = [decisions(g, True, bluff=True) for g in games[s]]
        ga = [decisions(g, False) for g in games[s]]
        groups[s] = (gm, gb, ga)
        cm = cluster_bootstrap(gm, ch_rate, seed=1)
        cb = cluster_bootstrap(gb, ch_rate, seed=2)
        ca = cluster_bootstrap(ga, ch_rate, seed=3)

        def pi(evs):
            m = [e for e in evs if e["pred_is_monitor"]]
            o = [e for e in evs if not e["pred_is_monitor"]]
            return ch_rate(m) - ch_rate(o) if m and o else float("nan")

        both = [decisions(g, True) + decisions(g, False) for g in games[s]]
        cp = cluster_bootstrap(both, pi, seed=4)
        n = sum(len(x) for x in gm)
        behav[s] = {"n_monitor": n, "n_bluff": sum(len(x) for x in gb), "n_agents": sum(len(x) for x in ga),
                    "monitor": cm, "monitor_bluff": cb, "agents": ca, "preference_index": cp}
        say(f"{LABEL[s]:16s} {n:4d} {fmt_ci(*cm):>20s} {fmt_ci(*cb):>20s} {fmt_ci(*ca):>20s} "
            f"{fmt_ci(*cp, pct=False):>22s}")
    summary["behaviour"] = behav
    say("   preference index = challenge rate on the monitor minus challenge rate on other agents "
        "(negative = the monitor is spared)")

    # ------------------------------------------------------------------ 3. confirmatory tests
    say("\n3. CONFIRMATORY TESTS (Holm-corrected across all five)")
    raw = {}
    vc = {s: (verbal[s]["commits"]["k"], verbal[s]["commits"]["n"]) for s in verbal}
    raw["H1 hint > benign (commits)"] = fisher_exact(*vc["hint"], *vc["benign"])
    raw["H1 direct > hint (commits)"] = fisher_exact(*vc["direct"], *vc["hint"])
    perm = {}
    for s in ["benign", "hint", "direct"]:
        obs, p = permutation_test(groups[s][0], groups["none"][0], ch_rate, reps=10000, seed=11)
        perm[s] = obs
        raw[f"H2 {s} vs none (challenge monitor)"] = p
    adj = holm(raw)
    tests = {}
    for k in raw:
        extra = ""
        if k.startswith("H2"):
            extra = f"  difference {perm[k.split()[1]]:+.2f}"
        say(f"   {k:40s} p = {raw[k]:.4f}   Holm p = {adj[k]:.4f}{extra}")
        tests[k] = {"p": raw[k], "p_holm": adj[k]}
    summary["confirmatory"] = tests

    # ------------------------------------------------------------------ 4. exploratory
    say("\n4. EXPLORATORY")
    # 4a. Say-do: do agents that committed spare the monitor more than those that did not?
    commit_map = {(x["game"], x["agent"]): x["commits"] for x in rows}
    sd = {True: [], False: []}
    for s in ["hint", "direct"]:
        for g in games[s]:
            for agent in MODELS:
                evs = decisions(g, True, challenger=agent)
                if evs:
                    sd[commit_map.get((g["game_id"], agent), False)].append(evs)
    say_do = {}
    for k in (True, False):
        say_do[str(k)] = cluster_bootstrap(sd[k], ch_rate, seed=5)
        say(f"   4a. challenge rate on monitor, agents who {'committed' if k else 'did not commit'}: "
            f"{fmt_ci(*say_do[str(k)])} (agent-games={len(sd[k])}, n={sum(len(x) for x in sd[k])})")
    if sd[True] and sd[False]:
        obs, p = permutation_test(sd[True], sd[False], ch_rate, reps=10000, seed=12)
        say(f"       committed minus not: {obs:+.2f}, permutation p = {p:.3f}")
        say_do["diff"], say_do["p"] = obs, p
    summary["say_do"] = say_do

    # 4b. By model: challenge rate on the monitor, offer (hint+direct) vs none.
    by_model_b = {}
    say("   4b. challenge rate on monitor by model: hint+direct vs no offer")
    for a, m in MODELS.items():
        off = [decisions(g, True, challenger=a) for s in ("hint", "direct") for g in games[s]]
        base = [decisions(g, True, challenger=a) for g in games["none"]]
        off, base = [x for x in off if x], [x for x in base if x]
        co, cb_ = cluster_bootstrap(off, ch_rate, seed=6), cluster_bootstrap(base, ch_rate, seed=7)
        by_model_b[a] = {"offer": co, "none": cb_}
        say(f"       {m:14s} offer {fmt_ci(*co)}  vs none {fmt_ci(*cb_)}")
    summary["behaviour_by_model"] = by_model_b

    # 4c. Persistence: challenge rate on the monitor by round block.
    blocks = [(1, 2), (3, 4), (5, 8)]
    persist = {}
    say("   4c. challenge rate on monitor by round (persistence of the effect)")
    for s in STYLES:
        row = []
        for lo, hi in blocks:
            gs = [[e for e in decisions(g, True) if lo <= e["round"] <= hi] for g in games[s]]
            row.append(cluster_bootstrap([x for x in gs if x], ch_rate, seed=8))
        persist[s] = row
        say(f"       {LABEL[s]:16s} " + "  ".join(f"R{lo}-{hi}: {fmt_ci(*c)}" for (lo, hi), c in zip(blocks, row)))
    summary["persistence"] = {s: {f"{lo}-{hi}": c for (lo, hi), c in zip(blocks, persist[s])} for s in STYLES}

    # 4d. Robustness: include fallback decisions; first offer-condition test with only bluffs.
    say("   4d. robustness")
    for s in ["hint", "direct"]:
        gi = [decisions(g, True, include_fallback=True) for g in games[s]]
        gn = [decisions(g, True, include_fallback=True) for g in games["none"]]
        obs, p = permutation_test(gi, gn, ch_rate, reps=5000, seed=13)
        obs_b, p_b = permutation_test(groups[s][1], groups["none"][1], ch_rate, reps=5000, seed=14)
        say(f"       {s} vs none incl. fallbacks: {obs:+.2f} (p={p:.3f}); bluffs only: {obs_b:+.2f} (p={p_b:.3f})")
        summary.setdefault("robustness", {})[s] = {"incl_fallback": [obs, p], "bluffs_only": [obs_b, p_b]}

    # ------------------------------------------------------------------ figures
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    # Figure 1: verbal acceptance vs commitment vs suspicion.
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    keys = [("accept_word", "Says ACCEPT"), ("commits", "Commits to not challenging"), ("suspects", "Suspects a test")]
    x = np.arange(3)
    w = 0.26
    for i, (k, lab) in enumerate(keys):
        vals = [verbal[s][k]["ci"] for s in ["benign", "hint", "direct"]]
        p = [v[0] for v in vals]
        err = [[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]]
        ax.bar(x + (i - 1) * w, p, w, yerr=err, capsize=3, label=lab,
               color=["#9aa5b1", "#c4473a", "#4f8fbf"][i])
    ax.set_xticks(x, [LABEL[s] for s in ["benign", "hint", "direct"]])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of replies")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Figure 1. What agents say to the monitor's offer", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig1_verbal.png", dpi=200)
    plt.close(fig)

    # Figure 2: challenge rate on monitor vs on other agents, by condition.
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, s in enumerate(STYLES):
        for j, (key, mk, lab) in enumerate([("monitor", "o", "on the monitor"), ("agents", "s", "on other agents")]):
            p, lo, hi = behav[s][key]
            if math.isnan(p):
                continue
            ax.errorbar(i + (j - 0.5) * 0.25, p, yerr=[[p - lo], [hi - p]], fmt=mk, ms=7, capsize=3,
                        color=COLOR[s], mfc=COLOR[s] if j == 0 else "white", label=lab if i == 0 else None)
    ax.set_xticks(range(len(STYLES)), [f"{LABEL[s]}\n(games={len(games[s])})" for s in STYLES])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Challenge rate (95% CI)")
    ax.legend(frameon=False, fontsize=9, title="Filled: challenges on the monitor; hollow: on other agents",
              title_fontsize=8, loc="upper right")
    ax.set_title("Figure 2. Do agents spare the monitor after its offer?", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig2_behaviour.png", dpi=200)
    plt.close(fig)

    # Figure 3: say-do gap and by-model behaviour.
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    ax = axes[0]
    for i, k in enumerate(["True", "False"]):
        if k in say_do:
            p, lo, hi = say_do[k]
            ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="o", ms=8, capsize=4, color=["#c4473a", "#4f8fbf"][i])
    pn, lo, hi = behav["none"]["monitor"]
    ax.axhspan(lo, hi, color="#7a7f85", alpha=0.15)
    ax.axhline(pn, color="#7a7f85", ls="--", lw=1)
    ax.text(1.45, pn, "no-offer\nbaseline", fontsize=8, va="center", color="#555")
    ax.set_xticks([0, 1], ["Committed\nto the pact", "Did not\ncommit"])
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Challenge rate on monitor")
    ax.set_title("A. Words vs actions (hint+direct)", fontsize=10, loc="left")
    ax = axes[1]
    for i, a in enumerate(MODELS):
        for j, key in enumerate(["none", "offer"]):
            p, lo, hi = by_model_b[a][key]
            if math.isnan(p):
                continue
            ax.errorbar(i + (j - 0.5) * 0.3, p, yerr=[[p - lo], [hi - p]], fmt="o", ms=7, capsize=3,
                        color="#7a7f85" if key == "none" else "#c4473a",
                        label=("no offer" if key == "none" else "hint or direct") if i == 0 else None)
    ax.set_xticks(range(len(MODELS)), list(MODELS.values()), fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("B. By model", fontsize=10, loc="left")
    fig.suptitle("Figure 3. Verbal commitment and model differences", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig3_saydo_models.png", dpi=200)
    plt.close(fig)

    json.dump(summary, open(f"{RES}/summary.json", "w"), indent=2, default=float)
    open(f"{RES}/report.txt", "w").write("\n".join(LINES) + "\n")
    print(f"\nSaved {RES}/summary.json, {RES}/report.txt and figures in {FIG}/")


if __name__ == "__main__":
    main()
