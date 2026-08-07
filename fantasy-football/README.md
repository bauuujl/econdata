# ffbets — fantasy football matchup analyzer

Works out which fantasy team is actually worth betting on, and by how much.

Point projections tell you who is favoured. They do not tell you whether the
price is any good. `ffbets` pulls your Sleeper league, projects every starter
under **your league's own scoring rules**, simulates the week 20,000 times,
and turns the resulting win probabilities into fair prices you can hold up
against a sportsbook.

No API key, no account, no dependencies — just Python 3.8+ and the standard
library.

```
$ ffbets analyze --league 1199943242200662016 --week 9 --odds book.json

==============================================================================
  The League of Ordinary Gentlemen  |  2025 Week 9
  PPR scoring  |  20,000 simulations  |  auto lineups
==============================================================================

MATCHUPS
------------------------------------------------------------------------------
> Oil of Olave                150.2 pts  +/- 26.8   win  62.8%   fair   -169
  Nabers suck my Penix        139.2 pts  +/- 28.5   win  37.2%   fair   +169
  margin: Oil of Olave by 12.0 (median); 80% of outcomes -37.3 to +58.7 for them

BEST BETS  (vs your sportsbook, 0.25 x Kelly)
------------------------------------------------------------------------------
SIDE                        BOOK    FAIR    MODEL    MARKET    EV/$1   STAKE
louisflores                 +180    +123    44.8%     34.4%   +0.254    3.5%
Oil of Olave                -120    -169    62.8%     56.7%   +0.151    4.5%
Nabers suck my Penix        +140    +169    37.2%     43.3%   -0.107       -

  -> Best value: louisflores at +180 (model 44.8% vs market 34.4%, EV +25.4% per $1)
     Book hold on that market: 4.0%
```

## Quick start

```bash
cd fantasy-football

# 1. See it work with no league and no network
python3 -m ffbets demo

# 2. Find your league ID from your Sleeper username
python3 -m ffbets leagues --user your_username

# 3. Analyze this week
python3 -m ffbets analyze --league 1199943242200662016
```

Optionally install it so `ffbets` is on your PATH:

```bash
pip install -e .
```

## Pricing your bets

Without prices, the tool ranks matchups by win probability — the "model
leans" view, which answers *which side is stronger*. Give it prices and it
answers the harder question: *which side is mispriced*.

```bash
# inline, repeatable
ffbets analyze --league <id> --odd "Oil of Olave=-120" --odd "louisflores=+180"

# or from a file
ffbets analyze --league <id> --odds book.json
```

`book.json` maps a team to a price. Keys match on team name, manager handle,
a unique fragment of either, or the roster ID. American (`-120`, `+180`) and
decimal (`1.83`) odds are both accepted:

```json
{
  "Oil of Olave": "-120",
  "Nabers suck my Penix": "+140",
  "louisflores": 180
}
```

When both sides of a matchup are priced, the book's margin is stripped out
before the comparison, so `MARKET` is the market's genuine opinion rather
than its opinion plus the vig. `EV/$1` is still computed against the real
price you would pay — the vig is a cost you actually bear, so a side can show
a positive edge and still be a losing bet.

`STAKE` is a fraction of bankroll from the Kelly criterion, scaled to a
quarter by default (`--kelly`). Full Kelly is optimal only if your
probabilities are exactly right, which they are not; quarter Kelly gives up
a little growth for a lot less ruin.

## Does it actually work?

Ask it. `backtest` replays finished weeks using only what was knowable
beforehand — that week's published projections and game logs from prior weeks
only — and grades itself:

```
$ ffbets backtest --league 1199943242200662016

  weeks 2-13  |  60 matchups graded

ACCURACY
------------------------------------------------------------------------------
  Favourite won            60.0% of the time
  Brier score              0.2385   (0.25 = coin flip, lower is better)
  Brier skill vs coinflip  +4.6%
  Log loss                 0.6704   (0.693 = coin flip)

TEAM SCORE PROJECTIONS
------------------------------------------------------------------------------
  Mean absolute error      24.5 points per team-week
  Bias                     -2.9 points  (under-projecting)

CALIBRATION  (does 70% mean 70%?)
------------------------------------------------------------------------------
  CONFIDENCE           N   PREDICTED    ACTUAL
  50-60%              25       54.9%     48.0%
  60-70%              11       63.6%     72.7%
  70-80%              17       74.7%     58.8%
  80-90%               7       84.3%     85.7%
```

Read that honestly: a real edge, but a small one on a small sample. Fantasy
weeks are genuinely noisy, and a single season is only ~60 matchups — far too
few to separate a 4% edge from luck. Run it on your own league before you
trust it, and size accordingly.

## How the model works

**Scoring.** Sleeper publishes each league's `scoring_settings` as a flat
`{stat: points}` map, and the stats and projection feeds use those same
keys — so scoring a player is a dot product. Reading the raw statline rather
than the pre-computed `pts_ppr` field means TE premium, 6-point passing TDs,
first-down bonuses and bucketed defensive scoring all just work. This is
checked against Sleeper's own published starter points: exact to the cent
across 300 starter-weeks.

**Means.** Each player's expected score is that week's Rotowire projection
(re-scored under your rules), blended 80/20 with their own recent game log
once there are at least three games to average. Projections already fold in
matchup and role; the game log is a minority vote against a projection that
has gone stale.

**Spread.** 14 points from a workhorse back and 14 from a boom-bust deep
threat are very different bets. Each player's week-to-week variance comes
from their own game log, shrunk toward a positional prior worth four
pseudo-games — so a receiver with two games played is treated as a typical
receiver, and one with twelve is treated as himself. Weeks a player missed
are excluded rather than counted as zeros, which would badly overstate a
healthy starter's spread.

**Availability.** Questionable tags cost 25% of a player's expected output,
doubtful 70%, and Out/IR/bye zero it. Byes are detected from the schedule,
not assumed.

**Simulation.** 20,000 draws of the whole week. Two choices matter:

- *Skew.* Scores are drawn from a lognormal fitted to each player's mean and
  standard deviation. Fantasy scoring is bounded near zero with a long right
  tail; a normal distribution would hand out negative receiving lines and
  understate ceilings.
- *Correlation.* Starters are not independent. A two-level factor model links
  them: a shared per-game factor (a shootout lifts both offences), a
  per-offence factor inside it (a quarterback and his top receiver rise
  together), and player noise on top. Defences load *negatively* on the
  offence they face. Independent draws would understate how often a big
  underdog wins — which is precisely where betting value lives.

**Lineups.** By default the tool uses the lineup each manager actually
submitted, and falls back to the best legal lineup for anyone who has not set
theirs yet. `--lineup optimal` assumes everyone plays their best lineup;
`--lineup declared` never substitutes. Slots are filled most-restrictive
first, so a lone quarterback is never burned in a SUPER_FLEX.

## Commands

| Command | What it does |
| --- | --- |
| `leagues --user NAME` | Look up your league IDs from a Sleeper username |
| `analyze --league ID` | Simulate a week and price every side |
| `backtest --league ID` | Grade the model against weeks already played |
| `demo` | Run the whole pipeline on synthetic data, no network |

Useful flags for `analyze`:

| Flag | Default | Effect |
| --- | --- | --- |
| `--week N` | current NFL week | Which week to analyze |
| `--sims N` | 20000 | More simulations, less Monte Carlo noise |
| `--seed N` | random | Reproducible runs |
| `--detail` | off | Print every starting lineup with means and spreads |
| `--html PATH` | — | Write a self-contained HTML report |
| `--json` | off | Machine-readable output for your own analysis |
| `--lineup` | `auto` | `auto`, `declared` or `optimal` |
| `--correlation` | 0.35 | How tightly teammates move together (0–0.9) |
| `--projection-weight` | 0.80 | Projections vs a player's own game log |
| `--kelly` | 0.25 | Kelly multiplier for stake sizing |
| `--offline` | off | Use only cached data, never touch the network |

Every API response is cached under `~/.cache/ffbets` (override with
`--cache-dir`), so repeated runs cost no network at all and `--offline` keeps
working on a plane.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

89 tests, no network required.

## Limits worth knowing

- Projections come from a single source (Rotowire, via Sleeper). One source
  is one opinion; a consensus of several would be better.
- The correlation parameters are reasoned defaults, not fitted values. If you
  have the data to fit them on your league, do.
- Only head-to-head moneylines are priced. No spreads, totals or player props.
- The backtest above covers one league and one season. That is enough to show
  the model is not broken; it is nowhere near enough to prove an edge.
- Sleeper only. ESPN and Yahoo leagues would need their own clients, though
  everything downstream of `sleeper.py` is league-agnostic.

## A note on betting

This is a model, and models are wrong in ways their authors cannot see. A 65%
favourite still loses about one week in three, and the "edge" it reports is
only as good as the projections underneath it. Treat the output as one
opinion among several, never stake money you cannot afford to lose, and if
betting has stopped being fun, stop. In the US, 1-800-GAMBLER.
