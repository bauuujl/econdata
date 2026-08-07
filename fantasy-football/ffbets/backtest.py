"""Score the model against weeks that have already been played.

A win probability is a claim, and a completed season is the only honest way
to check it.  For each finished week this replays the model using only what
was knowable beforehand -- that week's published projections and the game
logs from earlier weeks -- and then compares the predictions with what
actually happened.

Three numbers matter:

  hit rate     how often the favourite won.  Easy to read, but a model that
               says 51% and a model that says 95% both count as one hit.
  Brier score  mean squared error of the probabilities.  Lower is better;
               0.25 is what you get by guessing 50% every time, so anything
               at or above that means the model adds nothing.
  calibration  when it says 70%, does it happen 70% of the time?  A model
               can rank matchups well and still be badly calibrated, and
               calibration is what betting actually pays for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .analyzer import build_lineups
from .projections import ProjectionBuilder
from .simulate import CorrelationModel, WeekSimulator, compare, summarize_team


@dataclass
class Prediction:
    week: int
    home: str
    away: str
    p_home: float
    projected_home: float
    projected_away: float
    actual_home: float
    actual_away: float

    @property
    def home_won(self):
        return self.actual_home > self.actual_away

    @property
    def favorite_won(self):
        if self.actual_home == self.actual_away:
            return None
        return self.home_won if self.p_home >= 0.5 else not self.home_won

    @property
    def confidence(self):
        return max(self.p_home, 1.0 - self.p_home)


@dataclass
class BacktestResult:
    league_name: str
    season: str
    weeks: list = field(default_factory=list)
    predictions: list = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def decided(self):
        """Predictions that ended in a win or a loss, not a tie."""
        return [p for p in self.predictions if p.favorite_won is not None]

    @property
    def hit_rate(self):
        decided = self.decided
        if not decided:
            return 0.0
        return sum(1 for p in decided if p.favorite_won) / len(decided)

    @property
    def brier(self):
        decided = self.decided
        if not decided:
            return 0.0
        return sum((p.p_home - (1.0 if p.home_won else 0.0)) ** 2 for p in decided) / len(
            decided
        )

    @property
    def log_loss(self):
        decided = self.decided
        if not decided:
            return 0.0
        total = 0.0
        for prediction in decided:
            probability = min(max(prediction.p_home, 1e-6), 1 - 1e-6)
            outcome = 1.0 if prediction.home_won else 0.0
            total -= outcome * math.log(probability) + (1 - outcome) * math.log(
                1 - probability
            )
        return total / len(decided)

    @property
    def skill(self):
        """Brier skill score against the 'always 50%' baseline of 0.25."""
        return 1.0 - self.brier / 0.25 if self.brier else 0.0

    @property
    def score_error(self):
        """(mean absolute error, bias) of projected team scores, in points."""
        errors = []
        for prediction in self.predictions:
            errors.append(prediction.projected_home - prediction.actual_home)
            errors.append(prediction.projected_away - prediction.actual_away)
        if not errors:
            return 0.0, 0.0
        return sum(abs(e) for e in errors) / len(errors), sum(errors) / len(errors)

    def calibration(self, edges=(0.5, 0.6, 0.7, 0.8, 0.9, 1.01)):
        """Predicted vs realised win rate, bucketed by confidence."""
        buckets = []
        low = edges[0]
        for high in edges[1:]:
            inside = [p for p in self.decided if low <= p.confidence < high]
            if inside:
                buckets.append(
                    {
                        "low": low,
                        "high": min(high, 1.0),
                        "count": len(inside),
                        "predicted": sum(p.confidence for p in inside) / len(inside),
                        "actual": sum(1 for p in inside if p.favorite_won) / len(inside),
                    }
                )
            low = high
        return buckets


def backtest_league(
    client,
    league_id,
    first_week=2,
    last_week=None,
    n_sims=4000,
    seed=1,
    history_weeks=8,
    projection_weight=0.80,
    correlation=None,
    verbose=False,
):
    league = client.league(league_id)
    season = str(league["season"])
    scoring = league["scoring_settings"]
    roster_positions = league.get("roster_positions") or []
    rosters = client.league_rosters(league_id)
    users = {user["user_id"]: user for user in client.league_users(league_id)}
    owner_of = {roster["roster_id"]: roster.get("owner_id") for roster in rosters}

    from .analyzer import team_label

    if last_week is None:
        last_week = (league.get("settings") or {}).get("playoff_week_start", 15) - 1

    try:
        player_db = client.players()
    except Exception:
        player_db = {}

    result = BacktestResult(league_name=league.get("name", "League"), season=season)

    for week in range(int(first_week), int(last_week) + 1):
        try:
            matchups = client.matchups(league_id, week)
        except Exception as exc:
            if verbose:
                print("  week %s unavailable (%s)" % (week, exc))
            continue
        # A week nobody has played yet has nothing to grade against.
        if not matchups or not any(entry.get("points") for entry in matchups):
            continue

        builder = ProjectionBuilder(
            client,
            scoring,
            season,
            week,
            projection_weight=projection_weight,
            history_weeks=history_weeks,
            player_db=player_db,
            verbose=False,
        )
        # Grade the lineups managers actually submitted, not idealised ones.
        built, _ = build_lineups(
            matchups, rosters, builder, roster_positions, mode="declared"
        )

        simulator = WeekSimulator(
            correlation=correlation or CorrelationModel(), n_sims=n_sims, seed=seed + week
        )
        ordered = [(roster_id, built[roster_id]["starters"]) for roster_id in built]
        totals = simulator.run(ordered)

        actual = {entry["roster_id"]: (entry.get("points") or 0.0) for entry in matchups}
        teams = {}
        for roster_id, payload in built.items():
            user = users.get(owner_of.get(roster_id))
            teams[roster_id] = summarize_team(
                roster_id,
                team_label(user, roster_id),
                (user or {}).get("display_name", "?"),
                payload["starters"],
                totals[roster_id],
            )

        grouped = {}
        for entry in matchups:
            grouped.setdefault(entry.get("matchup_id"), []).append(entry["roster_id"])

        counted = 0
        for roster_ids in grouped.values():
            if len(roster_ids) != 2:
                continue
            home, away = teams[roster_ids[0]], teams[roster_ids[1]]
            head_to_head = compare(None, home, away)
            result.predictions.append(
                Prediction(
                    week=week,
                    home=home.name,
                    away=away.name,
                    p_home=head_to_head.p_home,
                    projected_home=home.mean,
                    projected_away=away.mean,
                    actual_home=actual.get(home.roster_id, 0.0),
                    actual_away=actual.get(away.roster_id, 0.0),
                )
            )
            counted += 1

        if counted:
            result.weeks.append(week)
        if verbose:
            print("  week %s: %d matchups graded" % (week, counted))

    return result


def render_backtest(result):
    lines = []
    out = lines.append
    rule = "-" * 78

    out("=" * 78)
    out("  BACKTEST  |  %s  |  %s season" % (result.league_name, result.season))
    out("=" * 78)

    decided = result.decided
    if not decided:
        out("  No completed matchups to grade.")
        return "\n".join(lines)

    weeks = result.weeks
    out(
        "  weeks %s-%s  |  %d matchups graded"
        % (min(weeks), max(weeks), len(decided))
    )
    out("")
    out("ACCURACY")
    out(rule)
    out("  Favourite won            %.1f%% of the time" % (100 * result.hit_rate))
    out("  Brier score              %.4f   (0.25 = coin flip, lower is better)" % result.brier)
    out("  Brier skill vs coinflip  %+.1f%%" % (100 * result.skill))
    out("  Log loss                 %.4f   (0.693 = coin flip)" % result.log_loss)

    mae, bias = result.score_error
    out("")
    out("TEAM SCORE PROJECTIONS")
    out(rule)
    out("  Mean absolute error      %.1f points per team-week" % mae)
    out(
        "  Bias                     %+.1f points  (%s)"
        % (bias, "over-projecting" if bias > 0 else "under-projecting")
    )

    out("")
    out("CALIBRATION  (does 70% mean 70%?)")
    out(rule)
    out("  %-14s %7s %11s %9s" % ("CONFIDENCE", "N", "PREDICTED", "ACTUAL"))
    for bucket in result.calibration():
        out(
            "  %-14s %7d %10.1f%% %8.1f%%"
            % (
                "%.0f-%.0f%%" % (100 * bucket["low"], 100 * bucket["high"]),
                bucket["count"],
                100 * bucket["predicted"],
                100 * bucket["actual"],
            )
        )

    out("")
    out(rule)
    if result.skill > 0.02:
        verdict = "beats a coin flip -- the probabilities carry real information"
    elif result.skill > -0.02:
        verdict = "about as good as a coin flip -- do not bet on it"
    else:
        verdict = "worse than a coin flip on this sample -- do not bet on it"
    out("  Verdict: %s." % verdict)
    out(
        "  Fantasy weeks are genuinely noisy; a season is only ~%d matchups, so "
        "treat these figures as indicative." % len(decided)
    )
    return "\n".join(lines)
