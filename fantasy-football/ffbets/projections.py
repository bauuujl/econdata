"""Build a scoring distribution for every player on a roster.

A point projection is only half the story: 14 points from a workhorse back
and 14 points from a boom-bust deep threat are very different bets.  Each
player therefore gets three numbers:

  mean      expected points under the league's own scoring rules
  sd        week-to-week spread, from the player's own game log, pulled
            toward a positional prior when the sample is thin
  play_prob probability they are actually on the field (injury, bye)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .scoring import score_statline

# Coefficient of variation by position: how noisy a typical week is relative
# to the player's own average.  Receivers live and die by touchdowns and long
# catches, so they sit high; quarterbacks accumulate steadily, so they sit low.
POSITION_CV = {
    "QB": 0.34,
    "RB": 0.55,
    "WR": 0.62,
    "TE": 0.66,
    "K": 0.45,
    "DEF": 0.72,
}
DEFAULT_CV = 0.60

# Weight of the positional prior, expressed in pseudo-games.  With four
# games of real history a player's own variance and the prior count equally.
PRIOR_GAMES = 4.0

# Rough replacement-level output for a player with no projection and no game
# log.  Only ever reached by deep bench pieces, but it keeps the model from
# emitting zeros that would silently flatter a lineup.
REPLACEMENT_MEAN = {"QB": 11.0, "RB": 6.0, "WR": 6.0, "TE": 4.5, "K": 7.0, "DEF": 6.0}

# How likely a player listed with each designation is to suit up.
INJURY_PLAY_PROB = {
    "questionable": 0.75,
    "doubtful": 0.30,
    "out": 0.0,
    "ir": 0.0,
    "pup": 0.0,
    "sus": 0.0,
    "dnr": 0.0,
    "na": 0.0,
    "cov": 0.50,
}

MIN_SD = 1.0


@dataclass
class PlayerOutlook:
    """One player's projected scoring distribution for a single week."""

    player_id: str
    name: str
    position: str
    nfl_team: str = None
    opponent: str = None
    mean: float = 0.0
    sd: float = 0.0
    play_prob: float = 1.0
    games_used: int = 0
    source: str = "prior"
    injury_status: str = None
    on_bye: bool = False
    # How far apart the projection sources were, in points. Folded into `sd`,
    # and kept here so a disputed player can be spotted in the detail view.
    disagreement: float = 0.0
    contributions: dict = field(default_factory=dict)

    @property
    def expected(self):
        """Mean points once the chance of not playing is priced in."""
        return self.mean * self.play_prob

    @property
    def flags(self):
        marks = []
        if self.on_bye:
            marks.append("BYE")
        elif self.injury_status:
            marks.append(self.injury_status.upper()[:4])
        return ",".join(marks)


def play_probability(injury_status):
    if not injury_status:
        return 1.0
    return INJURY_PLAY_PROB.get(str(injury_status).strip().lower(), 1.0)


def shrunk_sd(mean, history, position):
    """Blend a player's observed spread with a positional prior.

    Returns (sd, games_used).  The prior is expressed as a coefficient of
    variation so it scales with the player's own workload -- a 20-point-a-week
    back should have a wider absolute spread than a 6-point-a-week backup.
    """
    prior_cv = POSITION_CV.get(position, DEFAULT_CV)
    prior_var = (prior_cv * max(mean, 1.0)) ** 2

    games = len(history)
    if games >= 2:
        observed_mean = sum(history) / games
        observed_var = sum((x - observed_mean) ** 2 for x in history) / (games - 1)
    else:
        observed_var = prior_var
        games = 0

    blended = (games * observed_var + PRIOR_GAMES * prior_var) / (games + PRIOR_GAMES)
    return max(math.sqrt(blended), MIN_SD), games


class ProjectionBuilder:
    """Assembles per-player outlooks for one league-week.

    Pulls the week's projections plus every completed week before it, scores
    all of it with the league's own rules, then combines the two into a mean
    and a spread per player.
    """

    def __init__(
        self,
        client,
        scoring_settings,
        season,
        week,
        season_type="regular",
        projection_weight=0.80,
        history_weeks=8,
        player_db=None,
        verbose=False,
        ensemble=None,
        score_calibration=None,
    ):
        self.client = client
        self.scoring = scoring_settings
        self.season = str(season)
        self.week = int(week)
        self.season_type = season_type
        self.projection_weight = projection_weight
        self.history_weeks = history_weeks
        self.verbose = verbose
        self.player_db = player_db or {}

        if ensemble is None:
            # The historical default: projections lead, recent form checks them.
            from .sources import build_ensemble

            ensemble = build_ensemble(
                "rotowire:%g,form:%g" % (projection_weight, 1.0 - projection_weight)
            )
        self.ensemble = ensemble
        # Corrections measured from past weeks: a systematic bias in the
        # projections, and a systematic understatement of how volatile a week
        # actually is. Applied per player, which is exact at the team level.
        self.score_calibration = score_calibration

        self.projections = {}  # player_id -> {"points","meta"}
        self.history = {}  # player_id -> [points, ...] most recent last
        self.teams_playing = set()
        # Fantasy points each defence has allowed by position, for the
        # opponent-adjusted source.
        self.allowed = {}  # nfl_team -> position -> [points allowed, ...]
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        self._load_projections()
        self._load_history()

    def _load_projections(self):
        try:
            rows = self.client.week_projections(
                self.season, self.week, self.season_type
            )
        except Exception as exc:  # the projections feed is undocumented
            if self.verbose:
                print("  projections unavailable (%s); using game logs only" % exc)
            return

        for row in rows or []:
            player_id = row.get("player_id")
            stats = row.get("stats") or {}
            if not player_id:
                continue
            meta = row.get("player") or {}
            team = row.get("team") or meta.get("team")
            opponent = row.get("opponent")
            # A row with a game_id is a scheduled game; teams absent from that
            # set are on bye.
            if row.get("game_id") and team:
                self.teams_playing.add(team)
            self.projections[player_id] = {
                "points": score_statline(stats, self.scoring),
                "has_stats": bool(stats) and stats.get("gp") is not None,
                "team": team,
                "opponent": opponent,
                "position": meta.get("position"),
                "name": ("%s %s" % (meta.get("first_name") or "", meta.get("last_name") or "")).strip(),
                "injury_status": meta.get("injury_status"),
            }

    def _load_history(self):
        first = max(1, self.week - self.history_weeks)
        for week in range(first, self.week):
            try:
                payload = self.client.week_stats(self.season, week, self.season_type)
            except Exception as exc:
                if self.verbose:
                    print("  week %s stats unavailable (%s)" % (week, exc))
                continue
            for player_id, stats in (payload or {}).items():
                # Only count weeks the player actually appeared in; folding in
                # bye weeks and inactives as zeros would wildly overstate the
                # spread of a healthy starter.
                if not stats or not stats.get("gp"):
                    continue
                self.history.setdefault(player_id, []).append(
                    score_statline(stats, self.scoring)
                )
                self._record_allowed(player_id, stats)

    def _record_allowed(self, player_id, stats):
        """Log how many points a defence gave up to each position.

        Only defensive statlines carry these keys, so this doubles as the
        test for whether the entry is a team defence.
        """
        from .sources import ALLOWED_KEYS

        if "fan_pts_allow_qb" not in stats:
            return
        for position, key in ALLOWED_KEYS.items():
            value = stats.get(key)
            if value is not None:
                self.allowed.setdefault(player_id, {}).setdefault(position, []).append(
                    float(value)
                )

    def defense_factor(self, team, position):
        """How generous `team` has been to `position`, relative to the league.

        Above 1 means they give up more than average and the matchup is
        favourable. Returns None when there is not enough history to say.
        """
        per_team = {}
        for defence, positions in self.allowed.items():
            values = positions.get(position) or []
            if values:
                per_team[defence] = sum(values) / len(values)
        if len(per_team) < 8 or team not in per_team:
            return None
        league_average = sum(per_team.values()) / len(per_team)
        if league_average <= 0:
            return None
        return per_team[team] / league_average

    # ------------------------------------------------------------------
    def outlook(self, player_id):
        """Build the scoring distribution for a single player."""
        projection = self.projections.get(player_id, {})
        database = self.player_db.get(player_id, {})
        history = self.history.get(player_id, [])

        position = (
            projection.get("position")
            or database.get("position")
            or (database.get("fantasy_positions") or [None])[0]
            or "FLEX"
        )
        name = (
            projection.get("name")
            or ("%s %s" % (database.get("first_name") or "", database.get("last_name") or "")).strip()
            or player_id
        )
        team = projection.get("team") or database.get("team")
        opponent = projection.get("opponent")
        injury = projection.get("injury_status") or database.get("injury_status")

        estimate = self.ensemble.combine(player_id, self) if self.ensemble else None
        if estimate is not None:
            mean = estimate.mean
            source = estimate.label
            disagreement = estimate.disagreement
            contributions = estimate.contributions
        else:
            # Nothing has an opinion: a deep bench piece nobody projects and
            # who has not played. Replacement level keeps the lineup honest.
            mean = REPLACEMENT_MEAN.get(position, 5.0)
            source = "replacement"
            disagreement = 0.0
            contributions = {}

        mean = max(mean, 0.0)
        sd, games_used = shrunk_sd(mean, history, position)
        # Sources disagreeing is real uncertainty about this player, on top of
        # their week-to-week variance. The two are near enough independent to
        # add in quadrature.
        if disagreement > 0:
            sd = math.sqrt(sd * sd + disagreement * disagreement)
        if self.score_calibration is not None:
            mean, sd = self.score_calibration.apply(mean, sd)

        on_bye = bool(self.teams_playing) and bool(team) and team not in self.teams_playing
        probability = 0.0 if on_bye else play_probability(injury)

        return PlayerOutlook(
            player_id=player_id,
            name=name,
            position=position,
            nfl_team=team,
            opponent=opponent,
            mean=mean,
            sd=sd,
            play_prob=probability,
            games_used=games_used,
            source=source,
            injury_status=injury,
            on_bye=on_bye,
            disagreement=disagreement,
            contributions=contributions,
        )

    def outlooks(self, player_ids):
        return [self.outlook(player_id) for player_id in player_ids if player_id and player_id != "0"]
