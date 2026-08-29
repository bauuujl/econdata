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

## ESPN leagues

Sleeper is the default. For ESPN, pass `--platform espn`, the season, and your
league URL:

```bash
ffbets --platform espn --season 2026 \
  --espn-s2 "$ESPN_S2" --swid "$ESPN_SWID" \
  analyze --league 'https://fantasy.espn.com/football/team?leagueId=1234567&seasonId=2026'
```

The whole URL works — the league id and season are pulled out of it — or pass
the bare id.

### You will need cookies

**Every ESPN league endpoint returns 401 without them, including leagues set
to "public".** This is not optional and there is no way around it; ESPN closed
the anonymous read path. To get them:

1. Open your league in a browser, logged in.
2. Open developer tools → Application (or Storage) → Cookies → `espn.com`.
3. Copy the values of `espn_s2` (a long string) and `SWID` (looks like
   `{XXXXXXXX-....}`, braces included).

Set them as environment variables so they stay out of your shell history:

```bash
export ESPN_S2='...'
export ESPN_SWID='{....}'
```

They expire every so often; when they do you get a clear error saying the
credentials were rejected rather than a confusing empty result.

### What differs from Sleeper

**Scoring is applied by the adapter, not the model.** ESPN lets a league score
the same statistic differently by position — tight-end premium is the common
case — which a flat `{stat: points}` map cannot express. So the ESPN client
applies the league's own `scoringItems` itself, position by position, and
hands the model an already-scored total. Custom scoring, TE premium and
6-point passing TDs are all handled; the reported scoring label reflects what
the league actually does.

**Byes are read, not inferred.** ESPN publishes the full NFL schedule and every
team's bye week, so a player on bye is known rather than guessed at from an
absence in a projections feed. Verified against the live 2026 schedule: all 16
Week 1 fixtures and all 32 bye weeks resolve correctly.

**One request covers the week.** ESPN returns settings, teams, rosters,
matchups and every rostered player's full stats array in a single call, so a
week costs one fetch rather than one per feed.

**No username lookup.** ESPN has no public equivalent of `ffbets leagues
--user`; get the league id from your browser's address bar.

### A caveat for a season that has not started

The model's spread estimates come from players' game logs. In Week 1 of a new
season there are none, so every player falls back to the positional prior and
the projections carry the whole forecast. That runs fine — it is tested — but
the win probabilities are at their least informative in Week 1 and get sharper
each week as real games accumulate. `backtest` needs completed weeks and will
say so politely if there are none yet.

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
beforehand and grades itself four ways.

```
$ ffbets backtest --league 1199943242200662016

  weeks 2-13  |  60 matchups graded  |  1.4 stacked starter pairs per roster

ACCURACY
  Favourite won            61.7% of the time  (+/- 6.3)
  Brier score              0.2341   (0.25 = coin flip, lower is better)
  Brier skill vs coinflip  +6.4%
  Team score MAE           24.3 points   (bias -1.9)

WHERE THE SCORE COMES FROM  (Murphy decomposition of Brier)
                                    RAW   CALIBRATED
  Brier                          0.2341       0.2341
    reliability (calibration)     0.0075       0.0075   lower is better
    resolution (discrimination)   0.0183       0.0183   higher is better
    uncertainty                  0.2489       0.2489   irreducible
```

Read that decomposition rather than the hit rate. **Resolution** is the
information the model actually carries; **reliability** is how much of it
overconfidence gives back. Here resolution (0.0183) is more than double
reliability (0.0075), which says the edge is real information partly wasted
by stating it too confidently -- not an artefact of sounding sure. Only the
second term is fixable by recalibration; the first has to be earned.

`--html` writes the same thing with a reliability plot, drawn as inline SVG
with no plotting library.

### How precise is that hit rate, really?

The ±6.3 points on the hit rate is `sqrt(p(1-p)/n)` with n=60. **It is a
property of the sample, not of the model** — no modelling change can shrink
it, and halving it needs 240 matchups. But you do not have to wait four
seasons to measure the model precisely, because a win/loss result throws away
almost everything the simulator produced. Grading the *whole* forecast
distribution against the real margin uses far more of each matchup:

```
DISTRIBUTIONAL FIT  (grades the whole forecast, not just the winner)
  Mean CRPS                22.74 points   (lower is better)
  Margin spread ratio      1.035  (+/- 0.092)   0.4 SE from perfect
  Team-score spread ratio  1.073  (+/- 0.065)
  PIT coverage             50% -> 52%,  80% -> 77%,  95% -> 95%
```

This is the most useful thing the tool now reports, and it changes the
verdict. **The margin distribution — the only thing a win probability depends
on — is 0.4 standard errors from correct, and PIT coverage lands within a
couple of points of nominal at every level.** The overconfidence that the
5-bucket reliability table appears to show is mostly small-sample noise: 12
matchups per bucket cannot distinguish 62% from 50%, but 60 continuous
margins can say the spread is right.

That also explains why Platt recalibration keeps failing out-of-sample. It is
correcting a distortion that is not really there.

### The one real defect it did find

Team *totals* were under-dispersed by about 10% while margins were fine. That
pattern has only one explanation: a shock common to every team in the week,
which inflates totals but cancels out of a head-to-head margin. Measuring it
directly found roughly a 9-point swing hitting all ten teams at once, worth
~10% of team-score error.

The simulator had per-game and per-offence factors but nothing above them, so
a `slate_share` level was added on top. It raises team-score spread while
provably leaving margins alone (both are asserted in the tests):

| | Team SD | Margin SD | corr(home, away) |
| --- | --- | --- | --- |
| no slate factor | 22.29 | 31.69 | −0.01 |
| slate 0.10 | 23.31 | 31.50 | +0.09 |
| slate 0.20 | 24.30 | 31.32 | +0.17 |

Effect on the backtest: Brier skill **+6.4% → +6.9%**, and the week-common
share of team error fell from 9.7% to 1%. Small, but it is the model getting
structurally righter rather than a post-hoc patch — and once it was in, the
`score` calibration below had almost nothing left to correct, which is the
outcome you want.

### Score calibration

A third recalibration option corrects the simulator instead of its output:
`mean_scale` for systematic bias and `dispersion` for systematic
over/under-confidence, both fitted on continuous team scores rather than
binary results — several times the information per season. It competes on the
same out-of-sample footing as Platt and isotonic and is used only if it wins.

### Recalibration

Isotonic and Platt maps are fitted **out-of-sample**, in folds grouped by
week so that two matchups from the same week never land on opposite sides of
the split. On this league both make things worse:

```
  METHOD          BRIER    LOG LOSS     IMPROVE  FIT
* none           0.2341      0.6601          --  no adjustment
  platt          0.2582      0.7193     -0.0241  slope 0.647 (overconfident), intercept -0.154
  isotonic       0.2857      1.5223     -0.0516  12 knots
```

That is worth sitting with. The Platt slope of 0.647 confirms the
overconfidence -- the model really does state 70% when it means something
nearer 63% -- but correcting it from 60 matchups costs more in fitting noise
than it recovers. Isotonic, with far more freedom, is worse still. **The
honest conclusion is not "recalibrate", it is "you do not have enough data to
recalibrate yet."** The machinery selects on out-of-sample Brier, so it
declines to use a map that does not earn its place, and `--save-calibrator`
writes an identity map rather than pretending.

Feed a real season or two into it and the picture may change. That is what
`--calibrator` is for: it applies a saved map to live probabilities *before*
they reach expected value and Kelly, which is the only place recalibration
changes what you bet.

### Closing line value

Win/loss records converge slowly. The hit rate above carries a ±6.3 point
standard error, which cannot distinguish a 3% edge from luck. Closing line
value converges far faster because it grades the price rather than the
result, so it is the metric this tool leads with:

```
CLOSING LINE VALUE
  Bets graded              50
  Mean CLV                 +13.09%  (+/- 2.76)
  Beat the close           78.0%  (+/- 5.9)
  Mean EV at close         +0.083 per $1 staked
```

**Read the caveat before the number.** Private fantasy leagues have no
sportsbook, so there is no real closing line for a past season, and those
figures come from a simulated market. They demonstrate that the pipeline
works; they are not evidence of an edge, and the tool says so in its own
output. Supply real prices and it becomes a genuine measurement:

```bash
ffbets backtest --league <id> --lines lines.json
```

```json
{
  "5": {
    "Oil of Olave":  {"open": "-115", "close": "-140"},
    "louisflores":   {"open": "+180", "close": "+155"}
  }
}
```

## Staking, and what miscalibration costs

Quarter Kelly is a guard against a bad probability, not a cure. `stress`
settles the question with a controlled experiment instead of a backtest,
because a *backtest cannot answer it* -- with a simulated market you have
already decided whether the model has an edge, and the bankroll comparison
just reports your assumption back to you.

So the truth is specified instead: a real win probability is drawn, the
market prices it exactly plus vig, and the bettor sees an overconfident
version and bets whenever the distortion says there is value. The bettor has
**no edge at all**. Every apparent one is the miscalibration talking.

```
$ ffbets stress

OVERCONFIDENCE 1.80x   (a true 60% is stated as 67%)
  POLICY                          MEDIAN  5th PCTL   MEAN DD     RUIN
  Full Kelly                       0.616     0.111     58.2%    41.2%
  Half Kelly                       0.853     0.424     34.3%    10.5%
  Quarter Kelly                    0.938     0.674     18.6%     0.0%
  Quarter + 30% market             0.966     0.806     10.9%     0.0%
  Quarter + 30% market + caps      0.967     0.823     10.1%     0.0%
  Flat 2%                          0.973     0.846      8.7%     0.0%
```

Full Kelly on probabilities stretched by 1.8x ruins **41% of seasons**.
Quarter Kelly survives but still bleeds, and still endures an 18.6% drawdown
on the way. Shrinking 30% toward the market roughly halves the drawdown for
a third of the loss. At 1.0x -- a perfectly calibrated bettor -- no policy
places a single bet, which is the correct answer and a useful check that the
experiment is set up right.

Both guards are available live:

```bash
ffbets analyze --league <id> --odds book.json --shrinkage 0.3 --cap 0.05
```

`--shrinkage` blends the model toward the market's de-vigged probability
before sizing; `--cap` bounds any single stake.

## Choosing a de-vig method

Which side looks like value depends on how you remove the book's margin, so
the method is selectable with `--devig`. On a -300 / +240 market:

| Method | Favourite | Longshot |
| --- | --- | --- |
| proportional | 71.83% | 28.17% |
| shin | 72.79% | 27.21% |
| additive | 72.79% | 27.21% |
| power | 73.31% | 26.69% |

Worth noting, because it is easy to have backwards: **proportional
*understates* favourites**, it does not overstate them. It strips margin in
proportion to each side's probability, so in absolute terms it takes the most
off the favourite -- 3.2 points here against the longshot's 1.2. Since books
empirically load more margin onto longshots, that is the wrong direction, and
it is why the default here is **Shin**, which infers the margin from an
insider-money model and lands between the two extremes. The methods disagree
by about 1.5 points of probability on this market and more as it gets
lopsided, which is exactly the range where a bet is marginal.

## Ensembling projection sources

One feed is one opinion. `--sources` takes a weighted committee:

```bash
ffbets analyze --league <id> --sources rotowire:0.6,form:0.2,opponent:0.2
```

| Source | What it is |
| --- | --- |
| `rotowire` | Sleeper's published projections, re-scored under league rules |
| `form` | exponentially-weighted recent scoring, half-life 3 games |
| `opponent` | recent form adjusted by how generous the opposing defence has been to that position |
| `file` | your own numbers, via `--projection-file` (JSON or CSV) |

Disagreement between sources is not averaged away -- it is folded into the
player's spread. Two sources saying 14 and one saying 6 describes a player we
do not really know, and that uncertainty belongs in the simulation, where it
correctly pulls extreme win probabilities back toward the middle.

One caution: Sleeper's projections endpoint accepts a `company` parameter and
will happily return HTTP 200 for `sportradar`, `fantasypros` or `espn`, but
only `rotowire` comes back with populated statlines. There is no second
provider hiding behind that parameter, and a 200 is not a second opinion.

## Fitting the correlations

The simulator's factor loadings started as reasoned defaults. `correlations`
replaces them with values fitted to real game logs:

```
$ ffbets correlations --league <id> --save corr.json

OBSERVED  (correlation of fantasy points, then of the normals beneath)
  KIND      PAIR           POINTS    LATENT       N
  same_team QB-WR1         +0.370    +0.398     334
  same_team QB-WR2         +0.408    +0.437     309
  same_team QB-TE1         +0.254    +0.279     303
  same_team QB-RB1         +0.073    +0.080     335
  same_team WR1-WR2        -0.021    -0.025     330
  opposing  QB-QB          +0.243    +0.253     154
  defense   DEF-QB         -0.313    -0.373     345
  defense   DEF-OFF        -0.313    -0.408     388

FITTED LOADINGS
  rho_QB      0.771
  rho_WR      0.453
  rho_RB      0.228
  rho_TE      0.192
  rho_K       0.050
  game_share  0.468
  defense     0.596
```

Those correlations of *points* are converted to correlations of the *normals*
beneath the lognormal before fitting -- the transform compresses correlation,
so skipping that step leaves the simulator systematically under-correlated.
The loadings then come out of an alternating least squares fit, which needs
no optimiser because every parameter is linear once the others are held fixed.

The numbers are recognisably football. Quarterbacks track their offence
hardest (0.771); running backs barely do (0.228), because game script pushes
carries up exactly when passing volume falls; and defences load hard against
the offence they face (0.596).

Two of them are more interesting than the fit:

- **QB-RB1 is +0.07** -- essentially independent, despite sharing an offence.
- **WR1-WR2 is -0.02** -- two receivers on the same team are *not* positively
  correlated at all, because competition for the same targets cancels the
  shared-offence effect.

A single shared factor per offence can only ever produce positive teammate
correlation, so that second one is **unrepresentable** in this model. It is
reported as a residual rather than absorbed silently into the loadings:

```
WORST MISFITS
  same_team WR1-WR2        -0.025    +0.206   -0.230
  same_team WR1-TE1        -0.067    +0.087   -0.154
```

### What the fitted values change: almost nothing

Pass them back with `--correlations corr.json` and the league-wide effect is
tiny -- team score spread moves +0.4%, upset probability +0.14 points. That is
not a bug, and the reason matters: **correlation can only act where two
starters share an offence**, and a typical fantasy roster has 10 starters
spread across 8 NFL teams. This league averages 1.4 stacked pairs per roster
out of 45 possible.

Where it does bite, it bites hard. Six starters from one offence:

| Loadings | Stacked roster SD | Diversified SD |
| --- | --- | --- |
| independent | 17.58 | 17.67 |
| defaults | 21.50 | 17.62 |
| fitted | 22.60 | 17.68 |

Fitted loadings make a heavy stack 28% more volatile than independence and 5%
more than the defaults, against a closed-form check of 22.83. So the parameters
matter for stacking decisions and barely at all for a diversified league --
which is worth knowing before you spend effort tuning them. `backtest` prints
the stack count alongside every result so the number is never read without it.

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
| `--platform espn` | Read an ESPN league instead (needs `--season` and cookies) |
| `analyze --league ID` | Simulate a week and price every side |
| `backtest --league ID` | Grade the model: accuracy, decomposition, recalibration, CLV, staking |
| `correlations --league ID` | Fit the simulator's factor loadings from real game logs |
| `stress` | What miscalibration costs each staking rule |
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
| `--sources` | `rotowire,form` | Projection committee, with optional weights |
| `--correlations` | — | Fitted loadings from the `correlations` command |
| `--calibrator` | — | Saved recalibration map, applied before EV and Kelly |
| `--shrinkage` | 0.0 | Weight on the market's probability when sizing |
| `--cap` | 0.25 | Ceiling on a single stake |
| `--devig` | `shin` | `shin`, `power`, `additive` or `proportional` |
| `--kelly` | 0.25 | Kelly multiplier for stake sizing |
| `--offline` | off | Use only cached data, never touch the network |

Every API response is cached under `~/.cache/ffbets` (override with
`--cache-dir`), so repeated runs cost no network at all and `--offline` keeps
working on a plane.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

295 tests, no network required.

## Limits worth knowing

- **No real market exists for private-league fantasy matchups.** The CLV
  machinery is real and works the moment you supply prices, but every CLV
  figure produced without `--lines` comes from a simulated market and proves
  the pipeline, not an edge.
- **60 matchups is not enough to recalibrate.** Both maps currently lose to
  leaving the probabilities alone. The overconfidence is real and measurable;
  the correction is not yet reliable.
- **The correlation model cannot represent negative teammate correlation.**
  A single shared factor per offence only produces positive pairs, so
  WR1-WR2 (-0.02 observed) and WR1-TE1 (-0.07) are unreachable. They surface
  as residuals. A compositional target-share factor would fix it and is not
  implemented.
- **Only one projection provider is genuinely available.** The ensemble takes
  as many sources as you like, but of the external feeds only Rotowire
  returns data; `form` and `opponent` are computed here, and `file` is yours.
- The stress test specifies its own truth. It shows what miscalibration
  costs a staking rule; it does not measure your miscalibration.
- Only head-to-head moneylines are priced. No spreads, totals or player props.
- Sleeper and ESPN. Yahoo would need its own client, though everything
  downstream of the platform adapter is site-agnostic.
- The ESPN adapter is tested end-to-end against a fixture built to the real
  API's shape, and its unauthenticated parts (schedule, byes, scoring maths)
  are verified against live data. The authenticated league path cannot be
  exercised without someone's cookies, so treat your first live run as the
  real test.

## A note on betting

This is a model, and models are wrong in ways their authors cannot see. A 65%
favourite still loses about one week in three, and the "edge" it reports is
only as good as the projections underneath it. Treat the output as one
opinion among several, never stake money you cannot afford to lose, and if
betting has stopped being fun, stop. In the US, 1-800-GAMBLER.
