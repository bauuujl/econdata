"""Ties the pieces together: league in, betting recommendations out."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import odds as odds_math
from .devig import hold_percentage, remove_vig
from .lineup import declared_lineup, optimal_lineup, starting_slots
from .staking import blended_probability
from .projections import ProjectionBuilder
from .scoring import scoring_label
from .simulate import (
    CorrelationModel,
    WeekSimulator,
    compare,
    standard_error,
    summarize_team,
)


@dataclass
class BetRecommendation:
    team: str
    opponent: str
    roster_id: object
    model_probability: float  # after recalibration, if a calibrator is loaded
    fair_american: int
    raw_probability: float = None  # straight from the simulator
    staked_probability: float = None  # after shrinkage toward the market
    # Everything below is only populated when the bet was priced.
    price_decimal: float = None
    price_american: int = None
    implied_probability: float = None
    devig_probability: float = None
    edge: float = None  # model probability minus the de-vigged market
    expected_value: float = None  # profit per $1 staked
    kelly: float = None  # fraction of bankroll
    market_hold: float = None
    error: float = 0.0  # Monte Carlo standard error on the probability

    @property
    def priced(self):
        return self.price_decimal is not None


@dataclass
class LeagueAnalysis:
    league_name: str
    season: str
    week: int
    scoring: str
    n_sims: int
    matchups: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    lineup_mode: str = "declared"
    notes: list = field(default_factory=list)
    kelly_multiplier: float = 0.25


def team_label(user, roster_id):
    """Prefer the manager's custom team name, fall back to their handle."""
    if not user:
        return "Roster %s" % roster_id
    metadata = user.get("metadata") or {}
    return metadata.get("team_name") or user.get("display_name") or "Roster %s" % roster_id


def resolve_week(client, league, requested):
    """Pick the week to analyze, defaulting to the live NFL week."""
    if requested:
        return int(requested), None

    try:
        state = client.nfl_state()
    except Exception:
        return 1, "could not read the NFL calendar; defaulting to week 1"

    if str(league.get("season")) == str(state.get("season")):
        week = state.get("display_week") or state.get("week") or 1
        return max(1, int(week)), None

    return 1, (
        "league is from the %s season but the NFL calendar reads %s; "
        "defaulting to week 1 (pass --week to pick another)"
        % (league.get("season"), state.get("season"))
    )


def build_lineups(matchups, rosters, builder, roster_positions, mode="auto"):
    """Turn each roster into a starting lineup of PlayerOutlooks.

    `mode` is 'declared' (trust the manager's submitted lineup), 'optimal'
    (assume everyone starts their best legal lineup), or 'auto' -- which uses
    the declared lineup when it looks complete and optimizes when it does
    not, since Sleeper carries a stale or half-empty lineup for weeks that
    have not been set yet.
    """
    slots = starting_slots(roster_positions)
    roster_by_id = {roster["roster_id"]: roster for roster in rosters}
    built = {}
    optimized = []

    for entry in matchups:
        roster_id = entry["roster_id"]
        roster = roster_by_id.get(roster_id, {})
        player_ids = entry.get("players") or roster.get("players") or []
        outlooks = builder.outlooks(player_ids)

        declared_ids = [pid for pid in (entry.get("starters") or []) if pid and pid != "0"]
        declared = declared_lineup(outlooks, declared_ids)

        use_optimal = mode == "optimal" or (
            mode == "auto" and len(declared) < len(slots)
        )
        if use_optimal:
            starters, _ = optimal_lineup(outlooks, roster_positions)
            optimized.append(roster_id)
        else:
            starters = declared

        built[roster_id] = {
            "starters": starters,
            "roster": outlooks,
            "optimized": use_optimal,
        }

    return built, optimized


def pair_matchups(matchups):
    """Group the flat matchup feed into {matchup_id: [roster_id, ...]}."""
    grouped = {}
    for entry in matchups:
        grouped.setdefault(entry.get("matchup_id"), []).append(entry["roster_id"])
    return grouped


def analyze_league(
    client,
    league_id,
    week=None,
    n_sims=20000,
    seed=None,
    lineup_mode="auto",
    projection_weight=0.80,
    history_weeks=8,
    correlation=None,
    use_player_db=True,
    book_odds=None,
    kelly_multiplier=0.25,
    ensemble=None,
    calibrator=None,
    devig_method="shin",
    shrinkage=0.0,
    exposure_cap=0.25,
    verbose=False,
):
    league = client.league(league_id)
    week, note = resolve_week(client, league, week)
    notes = [note] if note else []

    users = {user["user_id"]: user for user in client.league_users(league_id)}
    rosters = client.league_rosters(league_id)
    matchups = client.matchups(league_id, week)
    if not matchups:
        raise ValueError(
            "no matchups posted for week %s of %s" % (week, league.get("name"))
        )

    player_db = {}
    if use_player_db:
        try:
            player_db = client.players()
        except Exception as exc:
            notes.append("player database unavailable (%s); names may be sparse" % exc)

    builder = ProjectionBuilder(
        client,
        league["scoring_settings"],
        league["season"],
        week,
        projection_weight=projection_weight,
        history_weeks=history_weeks,
        player_db=player_db,
        ensemble=ensemble,
        verbose=verbose,
    )
    if not builder.projections:
        notes.append(
            "no projections published for this week; means come from game logs"
        )

    roster_positions = league.get("roster_positions") or []
    built, optimized = build_lineups(
        matchups, rosters, builder, roster_positions, lineup_mode
    )
    if optimized and lineup_mode == "auto":
        notes.append(
            "%d lineup(s) not set yet; used the optimal legal lineup for those"
            % len(optimized)
        )

    simulator = WeekSimulator(
        correlation=correlation or CorrelationModel(), n_sims=n_sims, seed=seed
    )
    ordered = [(roster_id, built[roster_id]["starters"]) for roster_id in built]
    totals = simulator.run(ordered)

    owner_of = {roster["roster_id"]: roster.get("owner_id") for roster in rosters}
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

    results = []
    for matchup_id, roster_ids in sorted(
        pair_matchups(matchups).items(), key=lambda item: (item[0] is None, item[0])
    ):
        if len(roster_ids) != 2:
            if len(roster_ids) == 1:
                notes.append(
                    "%s has no opponent this week (bye)"
                    % teams[roster_ids[0]].name
                )
            continue
        home, away = teams[roster_ids[0]], teams[roster_ids[1]]
        results.append(compare(matchup_id, home, away))

    analysis = LeagueAnalysis(
        league_name=league.get("name", "League"),
        season=str(league.get("season")),
        week=week,
        # An adapter that has already applied its platform's rules says so.
        scoring=league.get("scoring_label") or scoring_label(league["scoring_settings"]),
        n_sims=n_sims,
        matchups=results,
        lineup_mode=lineup_mode,
        notes=notes,
        kelly_multiplier=kelly_multiplier,
    )
    if calibrator is not None and getattr(calibrator, "name", "identity") != "identity":
        notes.append(
            "win probabilities recalibrated with a fitted %s map (%s)"
            % (calibrator.name, calibrator.description)
        )
    analysis.recommendations = build_recommendations(
        results,
        book_odds,
        n_sims,
        kelly_multiplier,
        calibrator=calibrator,
        devig_method=devig_method,
        shrinkage=shrinkage,
        exposure_cap=exposure_cap,
    )
    return analysis


# ----------------------------------------------------------------------
# betting
# ----------------------------------------------------------------------
def match_team(key, teams):
    """Resolve an odds-file key to a team: roster id, name, or handle."""
    text = str(key).strip().lower()
    for team in teams:
        if text == str(team.roster_id):
            return team
    for team in teams:
        if text in (team.name.lower(), team.owner.lower()):
            return team
    partial = [
        team
        for team in teams
        if text in team.name.lower() or text in team.owner.lower()
    ]
    return partial[0] if len(partial) == 1 else None


def build_recommendations(
    matchups,
    book_odds,
    n_sims,
    kelly_multiplier,
    calibrator=None,
    devig_method="shin",
    shrinkage=0.0,
    exposure_cap=0.25,
):
    """Price every side, and where the book disagrees, size the bet.

    The probability travels through three stages before it is allowed to
    size anything: the simulator's raw output, a fitted recalibration map,
    then shrinkage toward the market's own de-vigged opinion. Expected value
    and Kelly are computed from the last of those, because that is the number
    with the fewest known ways of being wrong.
    """
    teams = [team for matchup in matchups for team in (matchup.home, matchup.away)]
    priced = {}
    if book_odds:
        for key, value in book_odds.items():
            team = match_team(key, teams)
            if team is not None:
                priced[team.roster_id] = odds_math.parse_odds(value)

    recommendations = []
    for matchup in matchups:
        for team, opponent, probability in (
            (matchup.home, matchup.away, matchup.p_home),
            (matchup.away, matchup.home, matchup.p_away),
        ):
            # Ties push in most sportsbook rules, so the bettable probability
            # is renormalised over decisive outcomes only.
            decisive = matchup.p_home + matchup.p_away
            fair_probability = probability / decisive if decisive > 0 else 0.5
            fair_probability = min(max(fair_probability, 1e-4), 1 - 1e-4)

            raw_probability = fair_probability
            if calibrator is not None:
                fair_probability = min(
                    max(calibrator.predict(fair_probability), 1e-4), 1 - 1e-4
                )

            recommendation = BetRecommendation(
                team=team.name,
                opponent=opponent.name,
                roster_id=team.roster_id,
                model_probability=fair_probability,
                raw_probability=raw_probability,
                fair_american=odds_math.probability_to_american(fair_probability),
                error=standard_error(probability, n_sims),
            )

            price = priced.get(team.roster_id)
            if price:
                recommendation.price_decimal = price
                recommendation.price_american = odds_math.decimal_to_american(price)
                recommendation.implied_probability = 1.0 / price

                other = priced.get(opponent.roster_id)
                if other:
                    fair_pair = remove_vig([price, other], devig_method)
                    recommendation.devig_probability = fair_pair[0]
                    recommendation.market_hold = hold_percentage([price, other])
                    baseline = fair_pair[0]
                else:
                    baseline = recommendation.implied_probability

                staked = blended_probability(fair_probability, baseline, shrinkage)
                recommendation.staked_probability = staked

                recommendation.edge = fair_probability - baseline
                recommendation.expected_value = odds_math.expected_value(staked, price)
                recommendation.kelly = odds_math.kelly_fraction(
                    staked, price, multiplier=kelly_multiplier, cap=exposure_cap
                )

            recommendations.append(recommendation)

    if any(item.priced for item in recommendations):
        recommendations.sort(
            key=lambda item: (item.expected_value if item.priced else -99),
            reverse=True,
        )
    else:
        recommendations.sort(key=lambda item: item.model_probability, reverse=True)
    return recommendations
