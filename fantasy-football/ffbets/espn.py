"""An ESPN client shaped like the Sleeper one, so the model needs no changes.

Everything downstream of the platform client -- projections, the simulator,
calibration, betting -- works on a small vocabulary: a league with scoring
rules and roster slots, rosters of player IDs, weekly matchups, and per-week
statlines. This module speaks ESPN's v3 fantasy API and answers in exactly
that vocabulary, so `analyze_league` and `backtest_league` accept an
`EspnClient` wherever they accept a `SleeperClient`.

Three things differ from Sleeper and are worth knowing:

**Authentication.** Sleeper's read endpoints are open. ESPN's are not: every
league endpoint returns 401 without cookies, including leagues set to
"public". You need `espn_s2` and `SWID` from a logged-in browser session.
See `README.md` for how to find them.

**Scoring is position-aware.** ESPN leagues can score the same statistic
differently by position -- TE premium is the common case -- which a flat
{stat: points} dot product cannot express. So this adapter applies the
league's rules itself, using each player's position, and hands the model an
already-scored total. The model's own scoring step then multiplies that by
one, which keeps a single code path for both platforms.

**Byes and opponents are published.** ESPN's pro-team endpoint gives the
real schedule and bye week for every NFL team, so the model no longer has to
infer a bye from a player's absence in the projections feed.
"""

from __future__ import annotations

import gzip
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
USER_AGENT = "Mozilla/5.0 (compatible; ffbets/1.1)"

TTL_STATIC = 7 * 24 * 3600
TTL_LEAGUE = 6 * 3600
TTL_LIVE = 15 * 60

# The single synthetic statistic the adapter emits. The league's real rules
# have already been applied by the time the model sees it.
APPLIED_KEY = "espn_points"
APPLIED_SCORING = {APPLIED_KEY: 1.0}

# ESPN's `defaultPositionId`.
POSITION_BY_ID = {
    1: "QB",
    2: "RB",
    3: "WR",
    4: "TE",
    5: "K",
    7: "P",
    9: "DL",
    10: "DL",
    11: "LB",
    12: "CB",
    13: "S",
    14: "DB",
    16: "DEF",
}

# ESPN's `lineupSlotId`, translated into the slot names `lineup.py` knows.
SLOT_BY_ID = {
    0: "QB",
    1: "QB",
    2: "RB",
    3: "WRRB_FLEX",  # RB/WR
    4: "WR",
    5: "REC_FLEX",  # WR/TE
    6: "TE",
    7: "SUPER_FLEX",  # ESPN calls it OP, "offensive player"
    8: "DL",
    9: "DL",
    10: "LB",
    11: "DL",
    12: "DB",
    13: "DB",
    14: "DB",
    15: "IDP_FLEX",
    16: "DEF",
    17: "K",
    18: "P",
    19: "HC",
    20: "BN",
    21: "IR",
    22: "BN",
    23: "FLEX",  # RB/WR/TE
    24: "EDR",
    25: "IDP_FLEX",
}

# ESPN reports an injury designation per player; these are the ones that
# change whether they take the field.
INJURY_BY_STATUS = {
    "ACTIVE": None,
    "NORMAL": None,
    "QUESTIONABLE": "Questionable",
    "DOUBTFUL": "Doubtful",
    "OUT": "Out",
    "INJURY_RESERVE": "IR",
    "SUSPENSION": "Sus",
    "PROBABLE": None,
}


class EspnError(RuntimeError):
    """Raised when ESPN is unreachable, or refuses the credentials given."""


def _stat_entry_kind(entry):
    """(is_projection, is_weekly) for one entry in a player's stats array.

    ESPN encodes both in the entry: `statSourceId` 0 is what happened and 1
    is what was forecast, `statSplitTypeId` 0 is a season total and 1 is a
    single week.
    """
    return entry.get("statSourceId") == 1, entry.get("statSplitTypeId") == 1


class EspnClient:
    """Reads an ESPN league and answers in the Sleeper client's shape."""

    platform = "espn"

    def __init__(
        self,
        league_id=None,
        season=None,
        espn_s2=None,
        swid=None,
        cache_dir=None,
        offline=False,
        timeout=45,
        verbose=False,
    ):
        self.league_id = str(league_id) if league_id else None
        self.season = str(season) if season else None
        # Credentials may come from the environment so they never have to be
        # typed into a shell history.
        self.espn_s2 = espn_s2 or os.environ.get("ESPN_S2")
        self.swid = swid or os.environ.get("ESPN_SWID")
        if self.swid and not self.swid.startswith("{"):
            self.swid = "{%s}" % self.swid.strip("{}")

        if cache_dir is None:
            cache_dir = os.path.join(
                os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
                "ffbets",
                "espn",
            )
        self.cache_dir = cache_dir
        self.offline = offline
        self.timeout = timeout
        self.verbose = verbose
        os.makedirs(self.cache_dir, exist_ok=True)

        self._payloads = {}  # week -> league payload
        self._schedule = None

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------
    @property
    def authenticated(self):
        return bool(self.espn_s2 and self.swid)

    def _cache_path(self, key):
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return os.path.join(self.cache_dir, safe[:180] + ".json.gz")

    def _read_cache(self, key, ttl):
        path = self._cache_path(key)
        try:
            age = time.time() - os.path.getmtime(path)
        except OSError:
            return None
        if age > ttl and not self.offline:
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return None

    def _write_cache(self, key, payload):
        path = self._cache_path(key)
        try:
            temp = path + ".tmp"
            with gzip.open(temp, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temp, path)
        except OSError:
            pass

    def get(self, path, params=None, ttl=TTL_LIVE, cache_key=None, fantasy_filter=None):
        url = BASE + path
        if params:
            url += "?" + urllib.parse.urlencode(params, doseq=True)
        key = cache_key or url[len(BASE) :]

        cached = self._read_cache(key, ttl)
        if cached is not None:
            return cached
        if self.offline:
            raise EspnError(
                "offline mode: no cached response for %s (run once with network "
                "access to populate the cache)" % path
            )

        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if fantasy_filter:
            headers["x-fantasy-filter"] = json.dumps(
                fantasy_filter, separators=(",", ":")
            )
        if self.authenticated:
            headers["Cookie"] = "espn_s2=%s; SWID=%s" % (self.espn_s2, self.swid)

        if self.verbose:
            print("  fetching %s" % url)
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise EspnError(
                    "ESPN refused the request (HTTP %s). Every league endpoint "
                    "needs cookies, even for leagues marked public. Pass "
                    "--espn-s2 and --swid, or set ESPN_S2 and ESPN_SWID. %s"
                    % (
                        exc.code,
                        "The credentials supplied were rejected -- they may have "
                        "expired."
                        if self.authenticated
                        else "None were supplied.",
                    )
                ) from exc
            if exc.code == 404:
                raise EspnError(
                    "no such ESPN league or season: %s" % url
                ) from exc
            raise EspnError("%s returned HTTP %s" % (url, exc.code)) from exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise EspnError("could not reach %s: %s" % (url, exc)) from exc

        self._write_cache(key, payload)
        return payload

    # ------------------------------------------------------------------
    # the NFL, which needs no authentication
    # ------------------------------------------------------------------
    def schedule(self, season=None):
        """Team abbreviations, bye weeks and the week-by-week fixture list."""
        season = str(season or self.season)
        if self._schedule and self._schedule["season"] == season:
            return self._schedule

        payload = self.get(
            "/seasons/%s" % season,
            params={"view": "proTeamSchedules_wl"},
            ttl=TTL_STATIC,
            cache_key="espn_proteams_%s" % season,
        )
        teams = (payload.get("settings") or {}).get("proTeams") or []

        abbrev = {}
        byes = {}
        opponents = {}  # week -> {team_abbrev: opponent_abbrev}
        for team in teams:
            team_id = team.get("id")
            code = (team.get("abbrev") or "").upper()
            if not team_id or not code:
                continue
            abbrev[team_id] = code
            if team.get("byeWeek"):
                byes[code] = int(team["byeWeek"])

        for team in teams:
            for week, games in (team.get("proGamesByScoringPeriod") or {}).items():
                for game in games:
                    home = abbrev.get(game.get("homeProTeamId"))
                    away = abbrev.get(game.get("awayProTeamId"))
                    if not home or not away:
                        continue
                    slot = opponents.setdefault(int(week), {})
                    slot[home] = away
                    slot[away] = home

        self._schedule = {
            "season": season,
            "abbrev": abbrev,
            "byes": byes,
            "opponents": opponents,
        }
        return self._schedule

    def nfl_state(self):
        """Current season and week, read off the league itself."""
        try:
            payload = self._payload()
        except EspnError:
            return {"season": self.season, "week": 1, "display_week": 1}
        status = payload.get("status") or {}
        week = status.get("latestScoringPeriod") or payload.get("scoringPeriodId") or 1
        return {
            "season": str(payload.get("seasonId") or self.season),
            "week": int(week),
            "display_week": int(week),
            "season_type": "regular",
        }

    # ------------------------------------------------------------------
    # the league
    # ------------------------------------------------------------------
    def _payload(self, week=None, league_id=None):
        """One league fetch carries settings, teams, rosters and matchups.

        Rostered players arrive with their full stats array -- every week's
        actuals and projections -- so a single call per week covers both the
        lineup and all the history the model needs.
        """
        league_id = str(league_id or self.league_id)
        if not league_id:
            raise EspnError("no ESPN league id given")
        season = str(self.season or "")
        cache_key = "espn_league_%s_%s_%s" % (league_id, season, week or "current")

        if cache_key in self._payloads:
            return self._payloads[cache_key]

        params = [
            ("view", "mSettings"),
            ("view", "mTeam"),
            ("view", "mRoster"),
            ("view", "mMatchup"),
            ("view", "mMatchupScore"),
        ]
        if week:
            params.append(("scoringPeriodId", int(week)))

        payload = self.get(
            "/seasons/%s/segments/0/leagues/%s" % (season, league_id),
            params=params,
            ttl=TTL_LIVE,
            cache_key=cache_key,
        )
        self._payloads[cache_key] = payload
        return payload

    def league(self, league_id=None):
        """The league in Sleeper's shape, with ESPN's rules already applied."""
        payload = self._payload(league_id=league_id)
        settings = payload.get("settings") or {}
        roster_settings = settings.get("rosterSettings") or {}
        schedule_settings = settings.get("scheduleSettings") or {}

        slot_counts = roster_settings.get("lineupSlotCounts") or {}
        roster_positions = []
        for slot_id, count in sorted(slot_counts.items(), key=lambda kv: int(kv[0])):
            name = SLOT_BY_ID.get(int(slot_id), "BN")
            roster_positions.extend([name] * int(count))

        return {
            "league_id": str(payload.get("id") or league_id or self.league_id),
            "name": settings.get("name") or "ESPN League",
            "season": str(payload.get("seasonId") or self.season),
            "total_rosters": len(payload.get("teams") or []),
            "status": "in_season",
            "roster_positions": roster_positions,
            # The adapter has already scored every player, so the model's own
            # scoring step is the identity.
            "scoring_settings": dict(APPLIED_SCORING),
            "scoring_label": self.scoring_label(payload),
            "settings": {
                "playoff_week_start": schedule_settings.get("matchupPeriodCount", 14) + 1,
            },
            "espn_raw_settings": settings,
        }

    def scoring_label(self, payload=None):
        """A human label for the league's receiving and passing scoring."""
        payload = payload or self._payload()
        items = (
            ((payload.get("settings") or {}).get("scoringSettings") or {}).get(
                "scoringItems"
            )
            or []
        )
        by_stat = {int(item.get("statId", -1)): item for item in items}

        reception = float((by_stat.get(53) or {}).get("points", 0.0) or 0.0)
        if reception >= 1.0:
            base = "PPR"
        elif reception >= 0.4:
            base = "half-PPR"
        elif reception > 0:
            base = "%gpt-PPR" % reception
        else:
            base = "standard"

        extras = []
        if float((by_stat.get(4) or {}).get("points", 4.0) or 0.0) >= 6.0:
            extras.append("6pt pass TD")
        overrides = (by_stat.get(53) or {}).get("pointsOverrides") or {}
        # Position 4 is the tight end; a bigger reception value there is the
        # usual way a league expresses TE premium.
        if overrides.get("4") and float(overrides["4"]) > reception:
            extras.append("TE premium %g" % float(overrides["4"]))
        return base + (" (%s)" % ", ".join(extras) if extras else "")

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def _scoring_map(self, payload=None):
        """{statId: (points, {positionId: points})} from the league's rules."""
        payload = payload or self._payload()
        items = (
            ((payload.get("settings") or {}).get("scoringSettings") or {}).get(
                "scoringItems"
            )
            or []
        )
        rules = {}
        for item in items:
            stat_id = item.get("statId")
            if stat_id is None:
                continue
            overrides = {}
            for key, value in (item.get("pointsOverrides") or {}).items():
                try:
                    overrides[int(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            rules[int(stat_id)] = (float(item.get("points") or 0.0), overrides)
        return rules

    def score_stats(self, stats, position_id, rules):
        """Apply the league's rules to one raw ESPN statline.

        Position-aware, because ESPN allows a statistic to be worth different
        amounts to different positions.
        """
        if not stats:
            return 0.0
        total = 0.0
        for stat_id, value in stats.items():
            try:
                rule = rules.get(int(stat_id))
            except (TypeError, ValueError):
                continue
            if rule is None or value is None:
                continue
            points, overrides = rule
            total += float(value) * overrides.get(position_id, points)
        return round(total, 2)

    # ------------------------------------------------------------------
    # rostered players
    # ------------------------------------------------------------------
    def _players_by_week(self, week=None):
        """Every rostered player, scored week by week under league rules.

        Returns {player_id: {meta, "actual": {week: pts}, "projected": {week: pts}}}.
        """
        payload = self._payload(week=week)
        rules = self._scoring_map(payload)
        schedule = self.schedule(payload.get("seasonId"))
        season = int(payload.get("seasonId") or self.season or 0)

        players = {}
        for team in payload.get("teams") or []:
            entries = ((team.get("roster") or {}).get("entries")) or []
            for entry in entries:
                pool = entry.get("playerPoolEntry") or {}
                player = pool.get("player") or entry.get("player") or {}
                player_id = player.get("id")
                if player_id is None:
                    continue

                position_id = player.get("defaultPositionId")
                position = POSITION_BY_ID.get(position_id, "FLEX")
                team_code = schedule["abbrev"].get(player.get("proTeamId"))

                record = players.setdefault(
                    str(player_id),
                    {
                        "name": player.get("fullName") or str(player_id),
                        "position": position,
                        "position_id": position_id,
                        "team": team_code,
                        "injury_status": INJURY_BY_STATUS.get(
                            (player.get("injuryStatus") or "").upper(), None
                        ),
                        "eligible_slots": player.get("eligibleSlots") or [],
                        "actual": {},
                        "projected": {},
                    },
                )

                for stat in player.get("stats") or []:
                    if int(stat.get("seasonId") or season) != season:
                        continue
                    projected, weekly = _stat_entry_kind(stat)
                    if not weekly:
                        continue
                    period = stat.get("scoringPeriodId")
                    if not period:
                        continue
                    raw = stat.get("stats") or {}
                    if not raw:
                        continue
                    points = self.score_stats(raw, position_id, rules)
                    bucket = "projected" if projected else "actual"
                    record[bucket][int(period)] = points

        return players

    def players(self):
        """The player database, in the shape the model expects."""
        players = self._players_by_week()
        return {
            player_id: {
                "first_name": record["name"].split(" ")[0],
                "last_name": " ".join(record["name"].split(" ")[1:]),
                "position": record["position"],
                "fantasy_positions": [record["position"]],
                "team": record["team"],
                "injury_status": record["injury_status"],
            }
            for player_id, record in players.items()
        }

    # ------------------------------------------------------------------
    # the Sleeper-shaped surface the model consumes
    # ------------------------------------------------------------------
    def league_users(self, league_id=None):
        payload = self._payload(league_id=league_id)
        members = {m.get("id"): m for m in (payload.get("members") or [])}
        users = []
        for team in payload.get("teams") or []:
            owners = team.get("owners") or []
            owner_id = owners[0] if owners else "team-%s" % team.get("id")
            member = members.get(owner_id) or {}
            display = (
                member.get("displayName")
                or ("%s %s" % (member.get("firstName") or "", member.get("lastName") or "")).strip()
                or "Manager %s" % team.get("id")
            )
            users.append(
                {
                    "user_id": str(owner_id),
                    "display_name": display,
                    "metadata": {"team_name": _team_name(team)},
                }
            )
        return users

    def league_rosters(self, league_id=None):
        payload = self._payload(league_id=league_id)
        rosters = []
        for team in payload.get("teams") or []:
            entries = ((team.get("roster") or {}).get("entries")) or []
            owners = team.get("owners") or []
            rosters.append(
                {
                    "roster_id": team.get("id"),
                    "owner_id": str(owners[0]) if owners else "team-%s" % team.get("id"),
                    "players": [
                        str(e.get("playerId"))
                        for e in entries
                        if e.get("playerId") is not None
                    ],
                    "starters": [
                        str(e.get("playerId"))
                        for e in entries
                        if SLOT_BY_ID.get(e.get("lineupSlotId"), "BN")
                        not in ("BN", "IR")
                    ],
                    "settings": {
                        "wins": (team.get("record") or {}).get("overall", {}).get("wins", 0),
                        "losses": (team.get("record") or {}).get("overall", {}).get("losses", 0),
                    },
                }
            )
        return rosters

    def matchups(self, league_id=None, week=None):
        """One entry per team, in the flat shape Sleeper publishes."""
        if week is None:
            league_id, week = None, league_id  # called as matchups(week)
        payload = self._payload(week=week, league_id=league_id)
        week = int(week)

        # Lineups live on the roster, which the payload returns for the week
        # requested.
        lineups = {}
        for team in payload.get("teams") or []:
            entries = ((team.get("roster") or {}).get("entries")) or []
            starters, everyone = [], []
            for entry in entries:
                player_id = entry.get("playerId")
                if player_id is None:
                    continue
                everyone.append(str(player_id))
                if SLOT_BY_ID.get(entry.get("lineupSlotId"), "BN") not in ("BN", "IR"):
                    starters.append(str(player_id))
            lineups[team.get("id")] = (starters, everyone)

        results = []
        for index, game in enumerate(payload.get("schedule") or []):
            if int(game.get("matchupPeriodId") or 0) != week:
                continue
            for side in ("home", "away"):
                entry = game.get(side) or {}
                roster_id = entry.get("teamId")
                if roster_id is None:
                    continue
                starters, everyone = lineups.get(roster_id, ([], []))
                results.append(
                    {
                        "roster_id": roster_id,
                        "matchup_id": game.get("id", index),
                        "starters": starters,
                        "players": everyone,
                        "points": entry.get("totalPoints") or 0.0,
                        "starters_points": [],
                    }
                )
        return results

    def week_stats(self, season, week, season_type="regular"):
        """What each rostered player actually scored, under league rules."""
        players = self._players_by_week()
        week = int(week)
        stats = {}
        for player_id, record in players.items():
            points = record["actual"].get(week)
            if points is None:
                continue
            stats[player_id] = {"gp": 1, APPLIED_KEY: points}
        return stats

    def week_projections(self, season, week, season_type="regular", positions=None):
        """Projections for the week, in the shape the projections feed uses."""
        players = self._players_by_week(week=week)
        schedule = self.schedule(season)
        week = int(week)
        opponents = schedule["opponents"].get(week, {})

        rows = []
        for player_id, record in players.items():
            team = record["team"]
            opponent = opponents.get(team) if team else None
            points = record["projected"].get(week)
            first, _, last = record["name"].partition(" ")
            rows.append(
                {
                    "player_id": player_id,
                    "team": team,
                    "opponent": opponent,
                    # A player whose team has no fixture this week is on bye,
                    # which the schedule states outright rather than leaving to
                    # be inferred.
                    "game_id": "%s-%s" % (team, week) if opponent else None,
                    "stats": (
                        {"gp": 1, APPLIED_KEY: points} if points is not None else {}
                    ),
                    "player": {
                        "first_name": first,
                        "last_name": last,
                        "position": record["position"],
                        "team": team,
                        "injury_status": record["injury_status"],
                    },
                }
            )
        return rows


def _team_name(team):
    """ESPN has moved the team name between fields across seasons."""
    name = team.get("name")
    if name:
        return name
    parts = [team.get("location") or "", team.get("nickname") or ""]
    joined = " ".join(part for part in parts if part).strip()
    return joined or "Team %s" % team.get("id")


def parse_league_url(text):
    """Pull leagueId and seasonId out of a pasted ESPN URL.

    Accepts the whole address bar, which is what people actually have.
    """
    text = str(text).strip()
    if text.isdigit():
        return text, None
    parsed = urllib.parse.urlparse(text)
    query = urllib.parse.parse_qs(parsed.query)
    league = (query.get("leagueId") or [None])[0]
    season = (query.get("seasonId") or query.get("season") or [None])[0]
    if not league:
        for part in parsed.path.split("/"):
            if part.isdigit() and len(part) >= 4:
                league = part
                break
    if not league:
        raise ValueError("could not find a leagueId in %r" % text)
    return league, season
