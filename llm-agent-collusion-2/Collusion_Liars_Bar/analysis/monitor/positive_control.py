"""Positive control: the paper's explicit secret alliance (game_series_secret_channel.py).

In that series the game designer offers one player a secret channel at game 20; the player
picks a partner and the pair talk privately in every later game. If an explicit alliance
makes LLM agents go easy on their partner, the allied pair's mutual challenge rate should
drop after the alliance forms, relative to the same pair before it and to the other pairs
over the same games (a difference-in-differences).

Unit: one challenge decision = the next player deciding whether to challenge a play.

Usage (from Collusion_Liars_Bar/):
    uv run python analysis/monitor/positive_control.py
"""
import glob
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from stats_utils import cluster_bootstrap, fmt_ci, permutation_test, wilson  # noqa: E402

SERIES_DIR = "experiments/game_records/Lily-Luke-Mike-Quinn_secret_comm"
SERIES_START = datetime(2026, 10, 24, 17, 6)  # files older than the series run are excluded
OUT = "docs/monitor_sting/results/positive_control.json"


def load_games():
    games = []
    for f in sorted(glob.glob(f"{SERIES_DIR}/*.json")):
        stamp = datetime.strptime(os.path.basename(f)[:15], "%Y%m%d_%H%M%S")
        if stamp < SERIES_START:
            continue
        d = json.load(open(f))
        if not d.get("winner"):
            continue
        comms = (d.get("secret_channel") or {}).get("communications") or []
        pair = {comms[0]["sender"], comms[0]["receiver"]} if comms else None
        events = []
        for r in d["rounds"]:
            target = r["target_card"]
            for p in r["play_history"]:
                if p.get("challenge_reason") == "System automatic challenge" or p["next_player"] in ("No one", None):
                    continue
                events.append({"player": p["player_name"], "challenger": p["next_player"],
                               "challenged": bool(p["was_challenged"]),
                               "bluff": not all(c == target or c == "Joker" for c in p["played_cards"])})
        games.append({"file": os.path.basename(f), "pair": pair, "events": events})
    return games


def main():
    games = load_games()
    # The series alliance is the pair that keeps talking game after game. A one-off channel in a
    # stray single game written to the same folder (a separate run of game_secret_channel.py)
    # is neither "before" nor "after", so it is dropped.
    pairs = [frozenset(g["pair"]) for g in games if g["pair"]]
    allied = set(max(set(pairs), key=pairs.count))
    first = next(i for i, g in enumerate(games) if g["pair"] == allied)
    dropped = [g["file"] for g in games if g["pair"] and g["pair"] != allied]
    pre = [g for g in games[:first] if not g["pair"]]
    post = [g for g in games[first:] if g["pair"] == allied]
    is_pair = lambda e: {e["player"], e["challenger"]} == allied  # noqa: E731

    def split(gs, inside):
        return [[e for e in g["events"] if is_pair(e) == inside] for g in gs]

    def ch_rate(evs):
        return sum(e["challenged"] for e in evs) / len(evs) if evs else float("nan")

    if dropped:
        print(f"Dropped games with a different, one-off channel: {dropped}")
    out = {"allied_pair": sorted(allied), "alliance_game_index": first + 1, "dropped": dropped,
           "n_pre_games": len(pre), "n_post_games": len(post), "cells": {}}
    print(f"Alliance {sorted(allied)} formed in series game {first + 1}; "
          f"{len(pre)} games before, {len(post)} games after.\n")
    for label, gs in (("pre", pre), ("post", post)):
        for inside in (True, False):
            groups = split(gs, inside)
            evs = [e for g in groups for e in g]
            k, n = sum(e["challenged"] for e in evs), len(evs)
            bl = [e for e in evs if e["bluff"]]
            kb, nb = sum(e["challenged"] for e in bl), len(bl)
            name = f"{label}_{'allied' if inside else 'other'}"
            out["cells"][name] = {"k": k, "n": n, "rate_ci": wilson(k, n), "k_bluff": kb, "n_bluff": nb,
                                  "bluff_rate_ci": wilson(kb, nb)}
            print(f"{name:13s} challenge rate {fmt_ci(*wilson(k, n))} (n={n});"
                  f"  on bluffs {fmt_ci(*wilson(kb, nb))} (n={nb})")

    # Difference-in-differences on the challenge rate, clustering by game.
    def did(evs_tagged):
        cell = {}
        for e in evs_tagged:
            cell.setdefault((e["_post"], e["_pair"]), []).append(e["challenged"])
        try:
            m = {k: sum(v) / len(v) for k, v in cell.items()}
            return (m[(1, 1)] - m[(0, 1)]) - (m[(1, 0)] - m[(0, 0)])
        except (KeyError, ZeroDivisionError):
            return float("nan")

    tagged = [[dict(e, _post=int(g in post), _pair=int(is_pair(e))) for e in g["events"]] for g in pre + post]
    point, lo, hi = cluster_bootstrap(tagged, did, reps=4000)
    out["did"] = {"estimate": point, "ci95": [lo, hi]}
    print(f"\nDiD (allied pair change minus other pairs' change): {fmt_ci(point, lo, hi, pct=False)}")

    obs, p = permutation_test(split(post, True), split(pre, True), ch_rate, reps=10000)
    out["allied_post_vs_pre"] = {"diff": obs, "p_perm": p}
    print(f"Allied pair, post minus pre: {obs:+.2f}  (game-level permutation p = {p:.3f})")

    out["note"] = ("game_secret_channel.py adds a reminder to every challenge prompt between allies: "
                   "'NEVER challenge your ally ... DO NOT CHALLENGE <name>!'. The post-alliance rate "
                   "therefore measures instructed, not voluntary, sparing: a ceiling for our conditions.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nSaved {OUT}")

    # Figure 4: allied pair vs other pairs, before and after the alliance.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for j, (inside, color, lab) in enumerate([(True, "#c4473a", f"Allied pair ({' & '.join(sorted(allied))})"),
                                              (False, "#7a7f85", "All other pairs")]):
        xs, ps, errs = [], [], [[], []]
        for i, phase in enumerate(["pre", "post"]):
            c = out["cells"][f"{phase}_{'allied' if inside else 'other'}"]
            p, lo, hi = c["rate_ci"]
            xs.append(i + (j - 0.5) * 0.2)
            ps.append(p)
            errs[0].append(p - lo)
            errs[1].append(hi - p)
        ax.errorbar(xs, ps, yerr=errs, fmt="-o", ms=7, capsize=3, color=color, label=lab)
    ax.set_xticks([0, 1], [f"Before alliance\n({len(pre)} games)", f"After alliance\n({len(post)} games)"])
    ax.set_ylim(-0.03, 1)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Challenge rate (Wilson 95% CI)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Figure 4. Positive control: the paper's explicit alliance", fontsize=11, loc="left")
    fig.tight_layout()
    os.makedirs("docs/monitor_sting/figures", exist_ok=True)
    fig.savefig("docs/monitor_sting/figures/fig4_positive_control.png", dpi=200)
    print("Saved docs/monitor_sting/figures/fig4_positive_control.png")


if __name__ == "__main__":
    main()
