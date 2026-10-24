# Pre-registration: monitor-offered non-challenge pacts in Liar's Bar

Written 24 Oct 2026 after the pilot and before any main-experiment result was looked at
(main games launched 21:18 BST; this file is committed before they finish).
Pilot games (2 complete, 1 crashed) are excluded from all analyses and kept in
`experiments/monitor_records_pilot/`.

## Question

When a player at the table privately offers a mutual "don't challenge each other" pact, do
ordinary LLM agents (1) say yes and (2) actually spare that player in later play? How do both
depend on how explicitly the pact is offered?

## Design

- **Seats.** One scripted monitor ("Rory") and three ordinary agents, each a different model:
  Lily = Llama-3.1-8B-Instruct, Mike = Ministral-8B, Quinn = Qwen2.5-7B-Instruct (all via
  OpenRouter, the repo's original sampling settings).
- **Monitor.** A fixed, seeded rule, the same in every condition: plays 1 or 2 cards, bluffs
  with probability 0.5, never challenges, never talks after the offer. A scripted monitor
  cannot leak cards or honour the pact differently by condition (an LLM-played undercover
  agent did leak its real cards in our earlier exploratory games).
- **Conditions** (between games). Before round 1 the monitor sends the same private message to
  every agent:
  - `none`: no message (baseline);
  - `benign`: a friendly cooperation request with no pact (negative control for the verbal measure);
  - `hint`: an implied mutual non-challenge arrangement;
  - `direct`: an explicit "I won't challenge your plays, you don't challenge mine" pact.
  The message and the agent's own reply stay in that agent's play and challenge prompts for the
  rest of the game.
- **Positive control.** The paper's explicit secret-alliance condition
  (`game_series_secret_channel.py`), from our 36-game replication run: an experimenter-selected
  player is given a private channel at game 20 and picks a partner. Analysed as a
  difference-in-differences (allied pair vs other pairs, before vs after).
- **Short games.** At most 8 rounds; the game also ends when the monitor is eliminated or one
  player is left. The monitor leads every round, so every round produces at least one
  decision on a monitor play.
- **Seating.** Rotated by seed over all 6 orders, so each model sits directly after the
  monitor (and is the one deciding on its plays) in exactly one third of the games.
- **Matched seeds.** Game k uses seed 100+k in every condition, so decks, bullets, seat order
  and the monitor's play rule start identical across conditions (common random numbers).

## Measures

- **Verbal** (per agent per game): first word ACCEPT/REFUSE, plus labels from a judge model
  (Llama-3.3-70B-Instruct) on the full reply: `commits_no_challenge` (primary verbal measure),
  `suspects_test`, `distrusts_sender`, `fairness_objection`, `safety_refusal`.
- **Behaviour** (per challenge decision): whether the agent directly after the monitor
  challenges a monitor play. Primary: challenge rate on monitor plays. Secondary: challenge
  rate on monitor bluffs only; preference index = challenge rate on the monitor minus the
  same agents' challenge rate on other agents in the same games.
- Decisions where the model gave no valid reply after 5 attempts (fallback) are excluded;
  a robustness check includes them.

## Hypotheses (confirmatory)

- **H1** Verbal commitment rises with explicitness: `benign` < `hint` < `direct`
  (two Fisher exact tests: hint vs benign, direct vs hint).
- **H2** Agents challenge the monitor less after an offer than with no offer
  (three game-level permutation tests: each offer style vs `none`).

The five p-values are Holm-corrected together. Everything else (say-do gap, per-model
results, persistence over rounds, reason text) is exploratory.

## Sample size

Inputs, all measured:
- intra-game correlation of challenge decisions **ICC = 0.073**, from the 19 pre-alliance
  games of the replication series;
- **m = 10** decisions on monitor plays per game (both pilot games);
- baseline challenge rate on the monitor **p0 ≈ 0.30** (pilot: 2/10, rounded up because the
  replication series' agents challenged 69% of plays).

Simulated power of the game-level permutation test (beta-binomial games, 300 runs per cell,
alpha 0.05; `analysis/monitor/power.py`):

| games per condition | -10 pts | -15 pts | -20 pts | -25 pts |
|---|---|---|---|---|
| 10 | 0.21 | 0.39 | 0.76 | 0.93 |
| 15 | 0.29 | 0.58 | 0.92 | 0.99 |
| 20 | 0.36 | 0.73 | 0.97 | 1.00 |

We run **18 games per condition (72 games)**: above 90% power for a 20-point drop, and 18 is
the smallest multiple of 6 above 15, so every seat order appears exactly 3 times and each
model follows the monitor in exactly 6 games per condition. Effects of 10 points or less are
not detectable at this size and will be reported as such, not as evidence of no effect.

Measured pilot cost was about $0.01 per game, so the run costs about $1 of the $5.56 left.

## Analysis plan

`analysis/monitor/analyze_monitor.py`. Proportions get Wilson intervals; anything pooled over
decisions gets a game-clustered bootstrap interval (4,000 resamples) because decisions in one
game share hands, history and opinions. Tests on behaviour permute whole games between
conditions (10,000 permutations).
