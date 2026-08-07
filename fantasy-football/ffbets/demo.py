"""A synthetic 10-team league, so the tool runs with no network and no league.

The numbers are invented but the code path is not: the demo builds real
PlayerOutlook objects and pushes them through the same simulator, summariser
and betting logic as a live league.
"""

from __future__ import annotations

import random

from .analyzer import LeagueAnalysis, build_recommendations
from .projections import PlayerOutlook, shrunk_sd
from .simulate import CorrelationModel, WeekSimulator, compare, summarize_team

TEAM_NAMES = [
    "Trust the Process",
    "Sunday Scaries",
    "Backfield in Motion",
    "Pylon Patrol",
    "Rushmore Faces",
    "Play Action Heroes",
    "Fourth and Goal",
    "Gridiron Gamblers",
    "The Waiver Wire",
    "Hail Mary Inc.",
]

NFL_TEAMS = [
    ("BUF", "MIA"), ("KC", "DEN"), ("SF", "SEA"), ("DAL", "PHI"), ("BAL", "CIN"),
    ("DET", "GB"), ("HOU", "IND"), ("LAC", "LV"), ("MIN", "CHI"), ("TB", "ATL"),
]

# (slot, mean range) for a standard QB/RB/RB/WR/WR/TE/FLEX/K/DEF lineup.
LINEUP_TEMPLATE = [
    ("QB", 14.0, 24.0),
    ("RB", 8.0, 21.0),
    ("RB", 6.0, 17.0),
    ("WR", 8.0, 22.0),
    ("WR", 6.0, 18.0),
    ("TE", 4.0, 14.0),
    ("WR", 5.0, 15.0),
    ("K", 6.0, 11.0),
    ("DEF", 3.0, 12.0),
]

FIRST = ["Jalen", "Marcus", "Devon", "Trey", "Kyren", "Bo", "Rome", "Chase",
         "Tank", "Xavier", "Drake", "Rashee", "Zay", "Jaxon", "Brock", "Puka"]
LAST = ["Harrison", "Nabers", "Bowers", "Odunze", "Worthy", "McConkey", "Bigsby",
        "Nacua", "Flowers", "Downs", "Coleman", "Legette", "Thomas", "Brooks"]


def build_demo_analysis(n_sims=20000, seed=7, book_odds=None, kelly_multiplier=0.25):
    generator = random.Random(seed)
    schedule = list(NFL_TEAMS)

    lineups = []
    labels = {}
    for index, name in enumerate(TEAM_NAMES):
        roster_id = index + 1
        starters = []
        for slot, low, high in LINEUP_TEMPLATE:
            home, away = generator.choice(schedule)
            on_home = generator.random() < 0.5
            team, opponent = (home, away) if on_home else (away, home)

            mean = generator.uniform(low, high)
            sd, _ = shrunk_sd(mean, [], slot)

            # A realistic sprinkling of injury designations.
            roll = generator.random()
            injury = "Questionable" if roll < 0.08 else ("Doubtful" if roll < 0.10 else None)
            play_prob = 0.75 if injury == "Questionable" else (0.30 if injury else 1.0)

            starters.append(
                PlayerOutlook(
                    player_id="%s-%s-%s" % (roster_id, slot, len(starters)),
                    name=(
                        "%s D/ST" % team
                        if slot == "DEF"
                        else "%s %s" % (generator.choice(FIRST), generator.choice(LAST))
                    ),
                    position=slot,
                    nfl_team=team,
                    opponent=opponent,
                    mean=mean,
                    sd=sd,
                    play_prob=play_prob,
                    games_used=0,
                    source="demo",
                    injury_status=injury,
                )
            )
        lineups.append((roster_id, starters))
        labels[roster_id] = name

    simulator = WeekSimulator(
        correlation=CorrelationModel(), n_sims=n_sims, seed=seed
    )
    totals = simulator.run(lineups)

    teams = {
        roster_id: summarize_team(
            roster_id, labels[roster_id], "demo", starters, totals[roster_id]
        )
        for roster_id, starters in lineups
    }

    results = []
    roster_ids = sorted(teams)
    for pair_index in range(0, len(roster_ids) - 1, 2):
        home = teams[roster_ids[pair_index]]
        away = teams[roster_ids[pair_index + 1]]
        results.append(compare(pair_index // 2 + 1, home, away))

    analysis = LeagueAnalysis(
        league_name="Demo League (synthetic data)",
        season="2025",
        week=8,
        scoring="PPR",
        n_sims=n_sims,
        matchups=results,
        lineup_mode="demo",
        notes=["synthetic data -- for trying the tool out, not for betting"],
        kelly_multiplier=kelly_multiplier,
    )
    analysis.recommendations = build_recommendations(
        results, book_odds, n_sims, kelly_multiplier
    )
    return analysis


def demo_book_odds(analysis, seed=11, vig=0.045):
    """Invent plausible book prices so the betting path has something to chew on.

    Prices are the model's own probabilities, nudged by a random mispricing
    and then loaded with a house margin -- which is roughly how a real market
    differs from a good model.
    """
    from .odds import decimal_to_american

    generator = random.Random(seed)
    prices = {}
    for matchup in analysis.matchups:
        drift = generator.uniform(-0.06, 0.06)
        decisive = matchup.p_home + matchup.p_away or 1.0
        for team, probability, direction in (
            (matchup.home, matchup.p_home, 1),
            (matchup.away, matchup.p_away, -1),
        ):
            fair = min(max(probability / decisive + direction * drift, 0.04), 0.96)
            prices[team.name] = decimal_to_american(1.0 / (fair * (1.0 + vig)))
    return prices
