"""Monte Carlo simulation of a fantasy week.

Adding up projections tells you who is favoured; it does not tell you by how
much you should care.  This module samples the whole week thousands of times
so that a win probability -- the thing you actually bet on -- falls out.

Two modelling choices matter more than the arithmetic:

Skew.  Fantasy scores are bounded below near zero and have a long right tail
(one broken tackle is a 60-yard touchdown).  Each player is drawn from a
lognormal fitted to their mean and standard deviation rather than a normal,
which would happily hand out negative receiving lines.

Correlation.  Starters are not independent.  A quarterback and their top
receiver rise together, a shootout lifts both offences in a game, and a
defence scores precisely when the offence it faces does not.  Independent
draws would understate how often a big underdog wins, which is exactly where
betting value hides.  Draws are therefore linked through a two-level factor
model: a shared per-game factor, a per-offence factor inside it, and player
noise on top.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class CorrelationModel:
    """Strength of the shared factors linking player outcomes."""

    # How much of a player's week is explained by their offence's day.
    offense: float = 0.35
    kicker: float = 0.20
    # Defences load negatively on the offence they are facing.
    defense: float = 0.30
    # Share of an offence's day driven by the game as a whole (pace, total)
    # rather than that offence alone -- this is what couples both sides of a
    # shootout together.
    game_share: float = 0.35


@dataclass
class TeamResult:
    roster_id: int
    name: str
    owner: str
    starters: list = field(default_factory=list)
    totals: list = field(default_factory=list)
    mean: float = 0.0
    sd: float = 0.0
    p10: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    win_probability: float = 0.0

    @property
    def projected(self):
        """Deterministic sum of starter expectations."""
        return sum(player.expected for player in self.starters)


@dataclass
class MatchupResult:
    matchup_id: object
    home: TeamResult
    away: TeamResult
    p_home: float = 0.0
    p_away: float = 0.0
    p_tie: float = 0.0
    mean_margin: float = 0.0
    median_margin: float = 0.0
    margin_p10: float = 0.0
    margin_p90: float = 0.0

    @property
    def favorite(self):
        return self.home if self.p_home >= self.p_away else self.away

    @property
    def underdog(self):
        return self.away if self.p_home >= self.p_away else self.home

    @property
    def favorite_probability(self):
        return max(self.p_home, self.p_away)

    @property
    def confidence(self):
        """How lopsided the matchup is, on a 0-1 scale."""
        return abs(self.p_home - self.p_away)

    @property
    def favorite_margin(self):
        """(median, p10, p90) of the margin from the favourite's point of view.

        Margins are stored home-minus-away; flipping the sign for an away
        favourite also swaps the tails, so the range always reads low-to-high
        in the favourite's favour.
        """
        if self.p_home >= self.p_away:
            return self.median_margin, self.margin_p10, self.margin_p90
        return -self.median_margin, -self.margin_p90, -self.margin_p10


def _lognormal_parameters(mean, sd):
    """Median and log-scale sigma of a lognormal with this mean and sd."""
    variance = sd * sd
    sigma_squared = math.log(1.0 + variance / (mean * mean))
    mu = math.log(mean) - 0.5 * sigma_squared
    return mu, math.sqrt(sigma_squared)


def _percentile(sorted_values, fraction):
    if not sorted_values:
        return 0.0
    position = fraction * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


class WeekSimulator:
    """Samples every roster in a week jointly, preserving correlations."""

    def __init__(self, correlation=None, n_sims=20000, seed=None):
        self.correlation = correlation or CorrelationModel()
        self.n_sims = int(n_sims)
        self.random = random.Random(seed)

    def _compile(self, lineups):
        """Flatten rosters into draw instructions, one per starter."""
        compiled = []
        teams = set()
        game_of_team = {}

        for roster_index, (_, starters) in enumerate(lineups):
            for player in starters:
                is_defense = player.position in ("DEF", "DST")
                driver = player.opponent if is_defense else player.nfl_team

                if player.nfl_team and player.opponent:
                    game_of_team[player.nfl_team] = tuple(
                        sorted((player.nfl_team, player.opponent))
                    )
                if driver:
                    teams.add(driver)

                if is_defense:
                    rho = -self.correlation.defense
                elif player.position == "K":
                    rho = self.correlation.kicker
                else:
                    rho = self.correlation.offense

                # A player who cannot post a positive mean (deep bench, or a
                # defence in a league that scores them harshly) gets a
                # truncated normal instead; the lognormal is undefined there.
                if player.mean > 0.05:
                    mu, sigma = _lognormal_parameters(player.mean, player.sd)
                    lognormal = True
                else:
                    mu, sigma = player.mean, player.sd
                    lognormal = False

                compiled.append(
                    (
                        roster_index,
                        driver,
                        rho,
                        math.sqrt(max(0.0, 1.0 - rho * rho)),
                        mu,
                        sigma,
                        player.play_prob,
                        lognormal,
                    )
                )

        return compiled, teams, game_of_team

    def run(self, lineups):
        """Simulate every roster. `lineups` is [(roster_id, [PlayerOutlook])].

        Returns {roster_id: [total, ...]} with one total per simulation.
        """
        compiled, teams, game_of_team = self._compile(lineups)
        games = sorted({game_of_team[team] for team in game_of_team})
        teams = sorted(teams)

        share = self.correlation.game_share
        game_weight = math.sqrt(share)
        team_weight = math.sqrt(1.0 - share)

        gauss = self.random.gauss
        uniform = self.random.random
        exp = math.exp

        totals = [[0.0] * self.n_sims for _ in lineups]

        for sim in range(self.n_sims):
            game_factor = {key: gauss(0.0, 1.0) for key in games}
            team_factor = {}
            for team in teams:
                own = gauss(0.0, 1.0)
                game_key = game_of_team.get(team)
                if game_key is None:
                    team_factor[team] = own
                else:
                    team_factor[team] = (
                        game_weight * game_factor[game_key] + team_weight * own
                    )

            for (
                roster_index,
                driver,
                rho,
                idiosyncratic,
                mu,
                sigma,
                play_prob,
                lognormal,
            ) in compiled:
                if play_prob <= 0.0:
                    continue
                if play_prob < 1.0 and uniform() >= play_prob:
                    continue

                shared = team_factor.get(driver, 0.0) if driver else 0.0
                z = rho * shared + idiosyncratic * gauss(0.0, 1.0)
                if lognormal:
                    score = exp(mu + sigma * z)
                else:
                    score = mu + sigma * z
                    if score < 0.0:
                        score = 0.0
                totals[roster_index][sim] += score

        return {lineups[index][0]: totals[index] for index, _ in enumerate(lineups)}


def summarize_team(roster_id, name, owner, starters, totals):
    ordered = sorted(totals)
    count = len(ordered)
    mean = sum(ordered) / count if count else 0.0
    variance = sum((x - mean) ** 2 for x in ordered) / (count - 1) if count > 1 else 0.0
    return TeamResult(
        roster_id=roster_id,
        name=name,
        owner=owner,
        starters=starters,
        totals=totals,
        mean=mean,
        sd=math.sqrt(variance),
        p10=_percentile(ordered, 0.10),
        p50=_percentile(ordered, 0.50),
        p90=_percentile(ordered, 0.90),
    )


def compare(matchup_id, home, away):
    """Head-to-head probabilities from two teams' simulated totals."""
    margins = [h - a for h, a in zip(home.totals, away.totals)]
    count = len(margins) or 1

    # Fractional scoring makes exact ties vanishingly rare in most leagues,
    # but integer-scoring leagues tie often enough to be worth counting.
    home_wins = sum(1 for margin in margins if margin > 1e-9)
    away_wins = sum(1 for margin in margins if margin < -1e-9)
    ties = count - home_wins - away_wins

    ordered = sorted(margins)
    result = MatchupResult(
        matchup_id=matchup_id,
        home=home,
        away=away,
        p_home=home_wins / count,
        p_away=away_wins / count,
        p_tie=ties / count,
        mean_margin=sum(margins) / count,
        median_margin=_percentile(ordered, 0.50),
        margin_p10=_percentile(ordered, 0.10),
        margin_p90=_percentile(ordered, 0.90),
    )
    home.win_probability = result.p_home
    away.win_probability = result.p_away
    return result


def standard_error(probability, n_sims):
    """Monte Carlo error on a win probability -- how much to trust the digits."""
    if n_sims <= 0:
        return 0.0
    return math.sqrt(max(probability * (1.0 - probability), 0.0) / n_sims)
