# Monitor-pact sting (AI Collusion Research Sprint, Oct 2026)

A scripted monitor seat in Liar's Bar privately offers every agent the same mutual
non-challenge pact (benign request / indirect hint / direct pact / no offer), and we measure
whether agents accept it and whether they then spare the monitor in play.

## What is new and what is pre-existing

**Pre-existing** (Zeng & Rudzicz, 2026, via the sprint starter repository
`poseidon-research/collusion_hackathon/llm-agent-collusion-2`): Liar's Bar, the player agent,
prompts, the secret-channel and secret-hint conditions, the API client, and the plotting scripts
in `analysis/visualization/` and `analysis/behavior/`.

**Written during the sprint:**
- `src/agents/monitor.py`: scripted monitor, offer texts, judge prompt, noted-player wrapper
- `src/game/game_monitor.py`: the monitor game variant and batch runner
- `analysis/monitor/`: power simulation, positive-control analysis, main analysis, statistics helpers
- `docs/monitor_sting/`: pre-registration, report draft, verified references, results and figures
- Fixes to pre-existing code: `src/llm/llm_client.py` (JSON clean-up no longer corrupts valid
  replies; quiet mode; stop on out-of-credit errors; short wait on rate limits; judge model added)
  and `src/agents/player.py` (reject plays with more copies of a card than the hand holds)

## Reproduce

From `Collusion_Liars_Bar/`, with an API key in the repo-root `compute.env`:

```bash
# 72 games, matched seeds (about 35 minutes with the four conditions in parallel; about $1)
for s in none benign hint direct; do uv run python src/game/game_monitor.py --style $s --games 18 --seed-start 100 & done; wait

uv run python analysis/monitor/power.py --m 10 --p0 0.30   # sample-size simulation
uv run python analysis/monitor/positive_control.py          # needs the secret-channel series
uv run python analysis/monitor/analyze_monitor.py           # all results and figures
```

## Files

| Path | Contents |
|---|---|
| `PREREGISTRATION.md` | Design, hypotheses, sample size; committed before the results were analysed |
| `REPORT_DRAFT.md` | Full report draft following the sprint template |
| `REFERENCES_VERIFIED.md` | Every reference and where it was checked |
| `results/report.txt` | Every reported number, readable |
| `results/summary.json`, `positive_control.json`, `power.json` | The same numbers, machine-readable |
| `figures/fig1`–`fig5` | Report figures |
| `figures/appendix_replication_*` | The repository's own plots of the original conditions |
