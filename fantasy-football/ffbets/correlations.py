"""Estimating the simulator's correlation structure from real game logs.

The simulator links players through a two-level factor model. Its loadings
started life as reasoned defaults; this module replaces them with numbers
fitted to what actually happened.

The procedure:

1. **Assign roles.** A "QB1" is not a player, it is a job. For each NFL team,
   the job goes to whoever did most of it over the season -- most pass
   attempts, most carries, most targets -- so the series stays stable when a
   backup spot-starts.

2. **Score and standardise.** Every role's weekly fantasy points are scored
   under the league's own rules, then converted to z-scores against that
   player's own mean and spread. Standardising is what makes a correlation
   between a 22-point quarterback and a 9-point tight end meaningful.

3. **Correlate.** Pearson correlations are pooled across all team-weeks for
   each pair of roles: teammates, opposing offences in the same game, and
   defences against the offence they faced.

4. **Undo the lognormal.** Those are correlations of *points*. The simulator
   correlates the *normal factors* underneath a lognormal, and the two are
   not equal -- the lognormal transform compresses correlation, more so for
   noisier positions. The inversion is exact and closed-form.

5. **Fit the loadings.** Each observed correlation is a product of model
   parameters: teammates give rho_p * rho_q, opposing offences give
   lambda * rho_p * rho_q, defences give -rho_def * rho_q. Every parameter
   enters linearly once the others are held fixed, so alternating least
   squares solves it in closed form per step, with no optimiser required.

The result is checked, not assumed: `tail_comparison` re-runs a real week
under the default and fitted loadings and reports how far the tail
probabilities move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .projections import DEFAULT_CV, POSITION_CV
from .scoring import score_statline
from .simulate import CorrelationModel

# The jobs we track. Each is filled by whoever leads the team in the usage
# statistic named alongside it.
ROLE_USAGE = {
    "QB": ("QB", "pass_att"),
    "RB1": ("RB", "rush_att"),
    "WR1": ("WR", "rec_tgt"),
    "WR2": ("WR", "rec_tgt"),
    "TE1": ("TE", "rec_tgt"),
    "K": ("K", "fga"),
}
ROLE_POSITION = {
    "QB": "QB",
    "RB1": "RB",
    "WR1": "WR",
    "WR2": "WR",
    "TE1": "TE",
    "K": "K",
    "DEF": "DEF",
    "OFF": "WR",  # the team-offence aggregate behaves like a mid-variance skill spot
}

OFFENSIVE_ROLES = ("QB", "RB1", "WR1", "WR2", "TE1", "K")

# Pairs worth estimating, and how the factor model says each one arises.
SAME_TEAM_PAIRS = [
    ("QB", "WR1"),
    ("QB", "WR2"),
    ("QB", "TE1"),
    ("QB", "RB1"),
    ("QB", "K"),
    ("WR1", "WR2"),
    ("WR1", "TE1"),
    ("RB1", "WR1"),
    ("RB1", "TE1"),
    ("WR2", "TE1"),
]
OPPOSING_PAIRS = [("QB", "QB"), ("QB", "WR1"), ("WR1", "WR1"), ("QB", "OFF")]
DEFENSE_PAIRS = [("DEF", "QB"), ("DEF", "WR1"), ("DEF", "OFF"), ("DEF", "RB1")]

MIN_OBSERVATIONS = 25


def sigma_log(position):
    """Log-scale spread implied by a position's coefficient of variation."""
    cv = POSITION_CV.get(position, DEFAULT_CV)
    return math.sqrt(math.log(1.0 + cv * cv))


def latent_correlation(observed, position_a, position_b):
    """Convert a correlation of points into one of the normals beneath them.

    For X = exp(mu + sigma Z) the relationship is closed-form:

        corr(X, Y) = (exp(rho sx sy) - 1) / sqrt((exp(sx^2)-1)(exp(sy^2)-1))

    Inverting it recovers rho. The transform always inflates -- a 0.30
    correlation in points implies a latent correlation above 0.30 -- so
    skipping this step would leave the simulator under-correlated.
    """
    sx, sy = sigma_log(position_a), sigma_log(position_b)
    scale = math.sqrt((math.exp(sx * sx) - 1.0) * (math.exp(sy * sy) - 1.0))
    inner = 1.0 + observed * scale
    if inner <= 1e-9:
        return -0.99
    rho = math.log(inner) / (sx * sy)
    return max(-0.99, min(0.99, rho))


def pearson(pairs):
    """Correlation of a list of (x, y) points."""
    count = len(pairs)
    if count < 3:
        return None
    mean_x = sum(x for x, _ in pairs) / count
    mean_y = sum(y for _, y in pairs) / count
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    var_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    var_y = sum((y - mean_y) ** 2 for _, y in pairs)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return None
    return covariance / math.sqrt(var_x * var_y)


@dataclass
class ObservedCorrelation:
    kind: str  # "same_team", "opposing", "defense"
    role_a: str
    role_b: str
    observed: float  # correlation of fantasy points
    latent: float  # correlation of the underlying normals
    count: int


@dataclass
class CorrelationFit:
    loadings: dict = field(default_factory=dict)  # position -> rho
    game_share: float = 0.35
    defense: float = 0.30
    observations: list = field(default_factory=list)
    weeks: list = field(default_factory=list)
    residual: float = 0.0

    def to_model(self):
        """The fitted CorrelationModel the simulator consumes."""
        return CorrelationModel(
            offense=self.loadings.get("WR", 0.35),
            kicker=self.loadings.get("K", 0.20),
            defense=self.defense,
            game_share=self.game_share,
            loadings=dict(self.loadings),
        )

    def implied(self, item):
        """What the fitted factor model says this pair's correlation is."""
        position_a = ROLE_POSITION.get(item.role_a, "WR")
        position_b = ROLE_POSITION.get(item.role_b, "WR")
        if item.kind == "same_team":
            return self.loadings.get(position_a, 0.35) * self.loadings.get(position_b, 0.35)
        if item.kind == "opposing":
            return (
                self.game_share
                * self.loadings.get(position_a, 0.35)
                * self.loadings.get(position_b, 0.35)
            )
        return -self.defense * self.loadings.get(position_b, 0.35)

    def residuals(self):
        """Per-pair misfit, worst first.

        Worth reading rather than skipping. A one-factor-per-offence model
        can only produce *positive* correlation between teammates, so any
        pair that is genuinely negative -- two receivers competing for the
        same targets -- is unrepresentable, and shows up here as a large
        residual rather than being silently absorbed into the loadings.
        """
        rows = []
        for item in self.observations:
            implied = self.implied(item)
            rows.append(
                {
                    "kind": item.kind,
                    "pair": "%s-%s" % (item.role_a, item.role_b),
                    "observed": item.latent,
                    "implied": implied,
                    "residual": item.latent - implied,
                    "count": item.count,
                }
            )
        return sorted(rows, key=lambda row: -abs(row["residual"]))

    def to_dict(self):
        return {
            "loadings": self.loadings,
            "game_share": self.game_share,
            "defense": self.defense,
            "weeks": self.weeks,
            "residual": self.residual,
            "observations": [
                {
                    "kind": item.kind,
                    "pair": "%s-%s" % (item.role_a, item.role_b),
                    "observed": round(item.observed, 4),
                    "latent": round(item.latent, 4),
                    "n": item.count,
                }
                for item in self.observations
            ],
        }


# ----------------------------------------------------------------------
# data collection
# ----------------------------------------------------------------------
def _week_frame(client, scoring, season, week, player_db, season_type):
    """Everything we need about one week: schedule, teams, positions, points."""
    try:
        stats = client.week_stats(season, week, season_type)
    except Exception:
        return None
    if not stats:
        return None

    team_of = {}
    position_of = {}
    schedule = {}
    try:
        rows = client.week_projections(season, week, season_type) or []
    except Exception:
        rows = []
    for row in rows:
        player_id = row.get("player_id")
        meta = row.get("player") or {}
        team = row.get("team") or meta.get("team")
        if not player_id or not team:
            continue
        team_of[player_id] = team
        if meta.get("position"):
            position_of[player_id] = meta["position"]
        if row.get("opponent") and row.get("game_id"):
            schedule[team] = row["opponent"]

    players = {}
    for player_id, statline in stats.items():
        if not statline or not statline.get("gp"):
            continue
        # Team defences are keyed by their own abbreviation.
        if player_id in schedule or (len(player_id) <= 3 and player_id.isalpha()):
            team, position = player_id, "DEF"
        else:
            database = player_db.get(player_id) or {}
            team = team_of.get(player_id) or database.get("team")
            position = position_of.get(player_id) or database.get("position")
        if not team or not position:
            continue
        players[player_id] = {
            "team": team,
            "position": position,
            "points": score_statline(statline, scoring),
            "stats": statline,
        }

    return {"week": week, "schedule": schedule, "players": players}


def _assign_roles(frames):
    """Give each team's jobs to whoever did most of the work."""
    usage = {}
    for frame in frames:
        for player_id, record in frame["players"].items():
            key = (record["team"], player_id)
            entry = usage.setdefault(
                key, {"position": record["position"], "usage": {}, "games": 0}
            )
            entry["games"] += 1
            for _, statistic in ROLE_USAGE.values():
                entry["usage"][statistic] = entry["usage"].get(statistic, 0.0) + float(
                    record["stats"].get(statistic, 0.0) or 0.0
                )

    roles = {}  # team -> role -> player_id
    teams = {team for team, _ in usage}
    for team in teams:
        candidates = [
            (player_id, entry)
            for (owner, player_id), entry in usage.items()
            if owner == team
        ]
        for role, (position, statistic) in ROLE_USAGE.items():
            ranked = sorted(
                (
                    (entry["usage"].get(statistic, 0.0), player_id)
                    for player_id, entry in candidates
                    if entry["position"] == position
                ),
                reverse=True,
            )
            ranked = [item for item in ranked if item[0] > 0]
            index = 1 if role == "WR2" else 0
            if len(ranked) > index:
                roles.setdefault(team, {})[role] = ranked[index][1]
        roles.setdefault(team, {})["DEF"] = team
    return roles


def _role_series(frames, roles):
    """Weekly points per (team, role), plus the team-offence aggregate."""
    series = {}  # (team, role) -> {week: points}
    for frame in frames:
        week = frame["week"]
        points_by_player = {
            player_id: record["points"] for player_id, record in frame["players"].items()
        }
        for team, assignments in roles.items():
            offense_total = 0.0
            offense_seen = 0
            for role, player_id in assignments.items():
                if player_id not in points_by_player:
                    continue
                points = points_by_player[player_id]
                series.setdefault((team, role), {})[week] = points
                if role in OFFENSIVE_ROLES and role != "K":
                    offense_total += points
                    offense_seen += 1
            if offense_seen >= 3:
                series.setdefault((team, "OFF"), {})[week] = offense_total
    return series


def _standardise(series):
    """Convert each role's weekly points into z-scores of its own history."""
    z_scores = {}
    for key, weekly in series.items():
        values = list(weekly.values())
        if len(values) < 3:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        spread = math.sqrt(variance)
        if spread < 1e-9:
            continue
        z_scores[key] = {
            week: (value - mean) / spread for week, value in weekly.items()
        }
    return z_scores


# ----------------------------------------------------------------------
# estimation
# ----------------------------------------------------------------------
def observe_correlations(frames, roles, z_scores):
    """Pool every role pair across team-weeks and correlate."""
    schedules = {frame["week"]: frame["schedule"] for frame in frames}
    teams = sorted(roles)
    observations = []

    def add(kind, role_a, role_b, pairs):
        correlation = pearson(pairs)
        if correlation is None or len(pairs) < MIN_OBSERVATIONS:
            return
        position_a = ROLE_POSITION.get(role_a, "WR")
        position_b = ROLE_POSITION.get(role_b, "WR")
        observations.append(
            ObservedCorrelation(
                kind=kind,
                role_a=role_a,
                role_b=role_b,
                observed=correlation,
                latent=latent_correlation(correlation, position_a, position_b),
                count=len(pairs),
            )
        )

    for role_a, role_b in SAME_TEAM_PAIRS:
        pairs = []
        for team in teams:
            left = z_scores.get((team, role_a), {})
            right = z_scores.get((team, role_b), {})
            for week in set(left) & set(right):
                pairs.append((left[week], right[week]))
        add("same_team", role_a, role_b, pairs)

    for role_a, role_b in OPPOSING_PAIRS:
        pairs = []
        for team in teams:
            for week, schedule in schedules.items():
                opponent = schedule.get(team)
                if not opponent or opponent <= team:
                    continue  # count each game once
                left = z_scores.get((team, role_a), {}).get(week)
                right = z_scores.get((opponent, role_b), {}).get(week)
                if left is not None and right is not None:
                    pairs.append((left, right))
        add("opposing", role_a, role_b, pairs)

    for role_a, role_b in DEFENSE_PAIRS:
        pairs = []
        for team in teams:
            for week, schedule in schedules.items():
                opponent = schedule.get(team)
                if not opponent:
                    continue
                left = z_scores.get((team, role_a), {}).get(week)
                right = z_scores.get((opponent, role_b), {}).get(week)
                if left is not None and right is not None:
                    pairs.append((left, right))
        add("defense", role_a, role_b, pairs)

    return observations


def fit_loadings(observations, iterations=400):
    """Alternating least squares for the factor loadings.

    Every constraint has the form `target = coefficient * parameter`, where
    the coefficient is built from the parameters currently held fixed. That
    makes each update a weighted least-squares solve in one unknown, which
    has a closed form, so the whole fit needs no optimiser and cannot wander.
    """
    loadings = {"QB": 0.35, "RB": 0.35, "WR": 0.35, "TE": 0.35, "K": 0.20}
    game_share = 0.35
    defense = 0.30

    def position(role):
        return ROLE_POSITION.get(role, "WR")

    def solve(coefficient_target_weight):
        """Weighted least squares for a single scalar parameter."""
        numerator = denominator = 0.0
        for coefficient, target, weight in coefficient_target_weight:
            numerator += weight * coefficient * target
            denominator += weight * coefficient * coefficient
        if denominator < 1e-12:
            return None
        return numerator / denominator

    for _ in range(iterations):
        previous = (dict(loadings), game_share, defense)

        # --- offensive loadings ---------------------------------------
        for name in loadings:
            constraints = []
            for item in observations:
                weight = float(item.count)
                pa, pb = position(item.role_a), position(item.role_b)
                if item.kind == "same_team":
                    # A pair of same-position roles (WR1 with WR2) makes the
                    # constraint rho^2 = target, which is not linear in rho.
                    # Holding one copy at its current value linearises it;
                    # the outer loop then iterates it to a fixed point.
                    if pa == name and pb == name:
                        constraints.append((loadings[name], item.latent, weight))
                    elif pa == name:
                        constraints.append((loadings[pb], item.latent, weight))
                    elif pb == name:
                        constraints.append((loadings[pa], item.latent, weight))
                elif item.kind == "opposing":
                    if pa == name and pb == name:
                        constraints.append(
                            (game_share * loadings[name], item.latent, weight)
                        )
                    elif pa == name:
                        constraints.append((game_share * loadings[pb], item.latent, weight))
                    elif pb == name:
                        constraints.append((game_share * loadings[pa], item.latent, weight))
                elif item.kind == "defense" and pb == name:
                    constraints.append((-defense, item.latent, weight))
            value = solve(constraints)
            if value is not None:
                loadings[name] = max(0.02, min(0.95, value))

        # --- share of an offence's day owned by the game ---------------
        constraints = []
        for item in observations:
            if item.kind != "opposing":
                continue
            product = loadings[position(item.role_a)] * loadings[position(item.role_b)]
            constraints.append((product, item.latent, float(item.count)))
        value = solve(constraints)
        if value is not None:
            game_share = max(0.0, min(0.95, value))

        # --- defensive loading ----------------------------------------
        constraints = []
        for item in observations:
            if item.kind != "defense":
                continue
            constraints.append(
                (-loadings[position(item.role_b)], item.latent, float(item.count))
            )
        value = solve(constraints)
        if value is not None:
            defense = max(0.0, min(0.95, value))

        moved = max(
            max(abs(loadings[k] - previous[0][k]) for k in loadings),
            abs(game_share - previous[1]),
            abs(defense - previous[2]),
        )
        if moved < 1e-9:
            break

    residual = 0.0
    total_weight = 0.0
    for item in observations:
        pa, pb = position(item.role_a), position(item.role_b)
        if item.kind == "same_team":
            implied = loadings[pa] * loadings[pb]
        elif item.kind == "opposing":
            implied = game_share * loadings[pa] * loadings[pb]
        else:
            implied = -defense * loadings[pb]
        residual += item.count * (implied - item.latent) ** 2
        total_weight += item.count
    residual = math.sqrt(residual / total_weight) if total_weight else 0.0

    return loadings, game_share, defense, residual


def estimate_correlations(
    client, scoring, season, weeks, player_db=None, season_type="regular", verbose=False
):
    """End-to-end: game logs in, fitted CorrelationModel out."""
    player_db = player_db or {}
    frames = []
    for week in weeks:
        frame = _week_frame(client, scoring, season, week, player_db, season_type)
        if frame and frame["players"]:
            frames.append(frame)
        if verbose:
            print(
                "  week %s: %d players"
                % (week, len(frame["players"]) if frame else 0)
            )
    if len(frames) < 4:
        raise ValueError(
            "need at least four completed weeks to estimate correlations, got %d"
            % len(frames)
        )

    roles = _assign_roles(frames)
    series = _role_series(frames, roles)
    z_scores = _standardise(series)
    observations = observe_correlations(frames, roles, z_scores)
    if not observations:
        raise ValueError("no role pair had enough overlapping weeks to correlate")

    loadings, game_share, defense, residual = fit_loadings(observations)
    return CorrelationFit(
        loadings=loadings,
        game_share=game_share,
        defense=defense,
        observations=observations,
        weeks=[frame["week"] for frame in frames],
        residual=residual,
    )


# ----------------------------------------------------------------------
# what it changes
# ----------------------------------------------------------------------
@dataclass
class TailComparison:
    label: str
    team_sd: float
    p01: float
    p05: float
    p95: float
    p99: float
    favorite_probability: float
    upset_probability: float
    blowout_probability: float
    close_game_probability: float


def tail_comparison(lineups, matchup_pairs, models, n_sims=20000, seed=5, blowout=30.0):
    """Re-simulate the same week under different correlation models.

    Correlation barely moves the mean -- it moves the tails, and the tails
    are where an underdog's win probability lives. This reports the pieces
    that actually shift.
    """
    from .simulate import WeekSimulator, _percentile

    results = []
    for label, model in models:
        totals = WeekSimulator(correlation=model, n_sims=n_sims, seed=seed).run(lineups)

        spreads = []
        percentiles = {"p01": [], "p05": [], "p95": [], "p99": []}
        for values in totals.values():
            ordered = sorted(values)
            mean = sum(ordered) / len(ordered)
            variance = sum((v - mean) ** 2 for v in ordered) / (len(ordered) - 1)
            spreads.append(math.sqrt(variance))
            percentiles["p01"].append(_percentile(ordered, 0.01))
            percentiles["p05"].append(_percentile(ordered, 0.05))
            percentiles["p95"].append(_percentile(ordered, 0.95))
            percentiles["p99"].append(_percentile(ordered, 0.99))

        favourite = []
        upset = []
        blowouts = []
        close = []
        for home_id, away_id in matchup_pairs:
            margins = [h - a for h, a in zip(totals[home_id], totals[away_id])]
            home_wins = sum(1 for margin in margins if margin > 0) / len(margins)
            top = max(home_wins, 1.0 - home_wins)
            favourite.append(top)
            upset.append(1.0 - top)
            blowouts.append(sum(1 for m in margins if abs(m) > blowout) / len(margins))
            close.append(sum(1 for m in margins if abs(m) < 5.0) / len(margins))

        def average(values):
            return sum(values) / len(values) if values else 0.0

        results.append(
            TailComparison(
                label=label,
                team_sd=average(spreads),
                p01=average(percentiles["p01"]),
                p05=average(percentiles["p05"]),
                p95=average(percentiles["p95"]),
                p99=average(percentiles["p99"]),
                favorite_probability=average(favourite),
                upset_probability=average(upset),
                blowout_probability=average(blowouts),
                close_game_probability=average(close),
            )
        )
    return results


def stack_intensity(lineups):
    """Average number of same-NFL-team starter pairs per roster.

    Worth reporting next to any correlation result. Correlation can only act
    where two starters share an offence, so a league of diversified rosters
    is nearly immune to the loadings no matter what they are -- and a reader
    who does not know the stack count will misread a small shift as a bug.
    """
    from collections import Counter

    if not lineups:
        return 0.0
    total = 0
    for _, starters in lineups:
        teams = Counter(player.nfl_team for player in starters if player.nfl_team)
        total += sum(count - 1 for count in teams.values() if count > 1)
    return total / len(lineups)
