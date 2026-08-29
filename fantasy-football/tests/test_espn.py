"""Tests for the ESPN adapter.

ESPN requires cookies for every league endpoint, so these run against a
fixture built to the shape the real API returns. The fixture is exercised
end-to-end through `analyze_league`, which is the claim that matters: an
`EspnClient` is accepted anywhere a `SleeperClient` is.
"""

import unittest

from ffbets.analyzer import analyze_league
from ffbets.espn import (
    APPLIED_KEY,
    POSITION_BY_ID,
    SLOT_BY_ID,
    EspnClient,
    parse_league_url,
)

# Slot ids: 0 QB, 2 RB, 4 WR, 6 TE, 23 FLEX, 16 D/ST, 17 K, 20 bench.
LINEUP_SLOT_COUNTS = {"0": 1, "2": 2, "4": 2, "6": 1, "23": 1, "16": 1, "17": 1, "20": 4}

# Full PPR, with a tight-end premium so the position-aware path is covered.
SCORING_ITEMS = [
    {"statId": 3, "points": 0.04},  # passing yards
    {"statId": 4, "points": 4.0},  # passing TDs
    {"statId": 20, "points": -2.0},  # interceptions
    {"statId": 24, "points": 0.1},  # rushing yards
    {"statId": 25, "points": 6.0},  # rushing TDs
    {"statId": 42, "points": 0.1},  # receiving yards
    {"statId": 43, "points": 6.0},  # receiving TDs
    {"statId": 53, "points": 1.0, "pointsOverrides": {"4": 1.5}},  # receptions, TE 1.5
]

# (player id, name, positionId, proTeamId, slot, weekly raw statline)
SQUAD = [
    (1, "Star Quarterback", 1, 28, 0, {"3": 260, "4": 2.0, "20": 0.6}),
    (2, "Lead Back", 2, 15, 2, {"24": 85, "25": 0.6, "53": 3, "42": 22}),
    (3, "Second Back", 2, 23, 2, {"24": 55, "25": 0.3, "53": 2, "42": 14}),
    (4, "Alpha Receiver", 3, 16, 4, {"53": 7, "42": 92, "43": 0.55}),
    (5, "Other Receiver", 3, 12, 4, {"53": 5, "42": 61, "43": 0.4}),
    (6, "Starting End", 4, 11, 6, {"53": 5, "42": 48, "43": 0.35}),
    (7, "Flex Receiver", 3, 17, 23, {"53": 4, "42": 52, "43": 0.3}),
    (8, "Home Defense", 16, 5, 16, {}),
    (9, "The Kicker", 5, 25, 17, {}),
    (10, "Bench Receiver", 3, 3, 20, {"53": 3, "42": 34, "43": 0.2}),
    (11, "Bench End", 4, 6, 20, {"53": 3, "42": 30, "43": 0.2}),
]


def stat_entries(season, raw, weeks=(1, 2, 3), projected_only=False):
    """A player's stats array: projections for every week, actuals for past ones."""
    entries = []
    for week in weeks:
        entries.append(
            {
                "id": "11%s%d" % (season, week),
                "statSourceId": 1,
                "statSplitTypeId": 1,
                "scoringPeriodId": week,
                "seasonId": int(season),
                "stats": dict(raw),
            }
        )
        if not projected_only and week < 3:
            # Actuals differ from the projection, which is the whole point.
            actual = {key: value * (1.15 if week == 1 else 0.8) for key, value in raw.items()}
            entries.append(
                {
                    "id": "01%s%d" % (season, week),
                    "statSourceId": 0,
                    "statSplitTypeId": 1,
                    "scoringPeriodId": week,
                    "seasonId": int(season),
                    "stats": actual,
                }
            )
    return entries


def build_payload(season=2026, teams=4, projected_only=False):
    """A league payload shaped like ESPN's, small enough to reason about."""
    team_entries = []
    members = []
    for index in range(teams):
        team_id = index + 1
        owner = "{OWNER-%d}" % team_id
        members.append({"id": owner, "displayName": "manager%d" % team_id})
        roster = []
        for player_id, name, position_id, pro_team, slot, raw in SQUAD:
            # Give each team its own player ids so rosters never overlap.
            unique_id = team_id * 100 + player_id
            roster.append(
                {
                    "playerId": unique_id,
                    "lineupSlotId": slot,
                    "playerPoolEntry": {
                        "player": {
                            "id": unique_id,
                            "fullName": "%s %d" % (name, team_id),
                            "defaultPositionId": position_id,
                            "proTeamId": pro_team,
                            "injuryStatus": "QUESTIONABLE" if player_id == 3 else "ACTIVE",
                            "eligibleSlots": [slot, 20, 21],
                            "stats": stat_entries(season, raw, projected_only=projected_only),
                        }
                    },
                }
            )
        team_entries.append(
            {
                "id": team_id,
                "name": "Team %d" % team_id,
                "owners": [owner],
                "record": {"overall": {"wins": 0, "losses": 0}},
                "roster": {"entries": roster},
            }
        )

    schedule = []
    for pair in range(teams // 2):
        schedule.append(
            {
                "id": pair + 1,
                "matchupPeriodId": 1,
                "home": {"teamId": pair * 2 + 1, "totalPoints": 0.0},
                "away": {"teamId": pair * 2 + 2, "totalPoints": 0.0},
            }
        )

    return {
        "id": 111222,
        "seasonId": season,
        "scoringPeriodId": 1,
        "status": {"latestScoringPeriod": 1},
        "members": members,
        "teams": team_entries,
        "schedule": schedule,
        "settings": {
            "name": "Fixture League",
            "rosterSettings": {"lineupSlotCounts": LINEUP_SLOT_COUNTS},
            "scoringSettings": {"scoringItems": SCORING_ITEMS},
            "scheduleSettings": {"matchupPeriodCount": 14},
        },
    }


def fixture_client(season=2026, **kwargs):
    """An EspnClient wired to the fixture, with no network anywhere."""
    client = EspnClient(league_id=111222, season=season, cache_dir="/tmp/ffbets-test")
    payload = build_payload(season=season, **kwargs)
    for key in ("espn_league_111222_%s_current" % season,):
        client._payloads[key] = payload
    # Every per-week key resolves to the same payload.
    for week in range(1, 19):
        client._payloads["espn_league_111222_%s_%d" % (season, week)] = payload
    client._schedule = {
        "season": str(season),
        "abbrev": {
            3: "CHI", 5: "CLE", 6: "DAL", 11: "IND", 12: "KC", 15: "MIA",
            16: "MIN", 17: "NE", 23: "PIT", 25: "SF", 28: "WSH",
        },
        "byes": {"MIA": 6, "PIT": 9},
        # Week 1 fixtures; MIN is deliberately absent so a bye can be detected.
        "opponents": {
            1: {
                "WSH": "PHI", "MIA": "LV", "PIT": "ATL", "KC": "DEN",
                "IND": "BAL", "NE": "SEA", "CLE": "JAX", "SF": "LAR",
                "CHI": "CAR", "DAL": "NYG",
            }
        },
    }
    return client


class MappingTests(unittest.TestCase):
    def test_offensive_slots_translate_to_names_the_model_knows(self):
        from ffbets.lineup import FLEX_ELIGIBILITY

        self.assertEqual(SLOT_BY_ID[0], "QB")
        self.assertEqual(SLOT_BY_ID[23], "FLEX")
        self.assertEqual(SLOT_BY_ID[7], "SUPER_FLEX")
        self.assertEqual(SLOT_BY_ID[20], "BN")
        for slot in ("FLEX", "SUPER_FLEX", "WRRB_FLEX", "REC_FLEX"):
            self.assertIn(slot, FLEX_ELIGIBILITY)

    def test_positions(self):
        self.assertEqual(POSITION_BY_ID[1], "QB")
        self.assertEqual(POSITION_BY_ID[16], "DEF")
        self.assertEqual(POSITION_BY_ID[5], "K")


class UrlTests(unittest.TestCase):
    def test_full_url_with_season(self):
        self.assertEqual(
            parse_league_url(
                "https://fantasy.espn.com/football/team?leagueId=123456&seasonId=2026&teamId=3"
            ),
            ("123456", "2026"),
        )

    def test_bare_id(self):
        self.assertEqual(parse_league_url("123456"), ("123456", None))

    def test_url_without_season(self):
        league, season = parse_league_url(
            "https://fantasy.espn.com/football/league?leagueId=987654"
        )
        self.assertEqual(league, "987654")
        self.assertIsNone(season)

    def test_junk_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_league_url("https://example.com/nothing-here")


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.client = fixture_client()
        self.rules = self.client._scoring_map(build_payload())

    def test_scores_a_receiver_under_full_ppr(self):
        # 7 receptions + 92 yards + 0.55 TDs
        points = self.client.score_stats({"53": 7, "42": 92, "43": 0.55}, 3, self.rules)
        self.assertAlmostEqual(points, 7 + 9.2 + 3.3, places=2)

    def test_tight_end_premium_is_applied(self):
        line = {"53": 6, "42": 60}
        receiver = self.client.score_stats(line, 3, self.rules)
        tight_end = self.client.score_stats(line, 4, self.rules)
        self.assertAlmostEqual(tight_end - receiver, 3.0, places=6)

    def test_negative_scoring_counts(self):
        self.assertAlmostEqual(
            self.client.score_stats({"3": 250, "4": 1, "20": 2}, 1, self.rules),
            10.0 + 4.0 - 4.0,
            places=2,
        )

    def test_unscored_stats_contribute_nothing(self):
        self.assertEqual(self.client.score_stats({"999": 50}, 3, self.rules), 0.0)

    def test_empty_statline(self):
        self.assertEqual(self.client.score_stats({}, 3, self.rules), 0.0)
        self.assertEqual(self.client.score_stats(None, 3, self.rules), 0.0)


class LeagueShapeTests(unittest.TestCase):
    def setUp(self):
        self.client = fixture_client()

    def test_roster_positions_are_expanded_in_slot_order(self):
        positions = self.client.league()["roster_positions"]
        self.assertEqual(positions.count("RB"), 2)
        self.assertEqual(positions.count("WR"), 2)
        self.assertEqual(positions.count("FLEX"), 1)
        self.assertEqual(positions.count("BN"), 4)
        self.assertEqual(positions[0], "QB")

    def test_scoring_settings_are_the_identity(self):
        # The adapter has already applied the league's real rules.
        league = self.client.league()
        self.assertEqual(league["scoring_settings"], {APPLIED_KEY: 1.0})

    def test_scoring_label_describes_the_real_rules(self):
        label = self.client.league()["scoring_label"]
        self.assertIn("PPR", label)
        self.assertIn("TE premium", label)

    def test_users_carry_team_names(self):
        users = self.client.league_users()
        self.assertEqual(len(users), 4)
        self.assertEqual(users[0]["metadata"]["team_name"], "Team 1")

    def test_rosters_split_starters_from_bench(self):
        roster = self.client.league_rosters()[0]
        self.assertEqual(len(roster["players"]), len(SQUAD))
        self.assertEqual(len(roster["starters"]), 9)
        self.assertNotIn("110", roster["starters"])  # bench receiver

    def test_matchups_pair_every_team_once(self):
        matchups = self.client.matchups(111222, 1)
        self.assertEqual(len(matchups), 4)
        ids = [entry["matchup_id"] for entry in matchups]
        self.assertEqual(sorted(ids), [1, 1, 2, 2])
        self.assertEqual(len(matchups[0]["starters"]), 9)


class StatFeedTests(unittest.TestCase):
    def setUp(self):
        self.client = fixture_client()

    def test_week_stats_are_pre_scored(self):
        stats = self.client.week_stats(2026, 1)
        self.assertTrue(stats)
        sample = next(iter(stats.values()))
        self.assertEqual(set(sample), {"gp", APPLIED_KEY})
        self.assertGreater(sample[APPLIED_KEY], 0)

    def test_a_week_with_no_actuals_is_empty(self):
        # The fixture only carries actuals for weeks 1 and 2.
        self.assertEqual(self.client.week_stats(2026, 3), {})

    def test_projections_carry_team_and_opponent(self):
        rows = {row["player_id"]: row for row in self.client.week_projections(2026, 1)}
        quarterback = rows["101"]  # team 1's QB, WSH
        self.assertEqual(quarterback["team"], "WSH")
        self.assertEqual(quarterback["opponent"], "PHI")
        self.assertIsNotNone(quarterback["game_id"])
        self.assertGreater(quarterback["stats"][APPLIED_KEY], 0)

    def test_a_team_with_no_fixture_is_flagged_as_a_bye(self):
        # MIN is missing from the week 1 schedule in the fixture.
        rows = {row["player_id"]: row for row in self.client.week_projections(2026, 1)}
        receiver = rows["104"]  # MIN
        self.assertEqual(receiver["team"], "MIN")
        self.assertIsNone(receiver["opponent"])
        self.assertIsNone(receiver["game_id"])

    def test_injury_status_is_translated(self):
        rows = {row["player_id"]: row for row in self.client.week_projections(2026, 1)}
        self.assertEqual(rows["103"]["player"]["injury_status"], "Questionable")
        self.assertIsNone(rows["101"]["player"]["injury_status"])

    def test_player_database_shape(self):
        database = self.client.players()
        entry = database["101"]
        self.assertEqual(entry["position"], "QB")
        self.assertEqual(entry["team"], "WSH")


class EndToEndTests(unittest.TestCase):
    """The claim: the model runs on ESPN with no changes of its own."""

    def test_analyze_runs_and_produces_probabilities(self):
        analysis = analyze_league(
            fixture_client(), 111222, week=1, n_sims=3000, seed=1, use_player_db=True
        )
        self.assertEqual(len(analysis.matchups), 2)
        for matchup in analysis.matchups:
            self.assertAlmostEqual(
                matchup.p_home + matchup.p_away + matchup.p_tie, 1.0, places=6
            )
            self.assertGreater(matchup.home.mean, 0)
        self.assertIn("PPR", analysis.scoring)

    def test_lineups_fill_every_scoring_slot(self):
        analysis = analyze_league(
            fixture_client(), 111222, week=1, n_sims=1000, seed=1
        )
        for matchup in analysis.matchups:
            for team in (matchup.home, matchup.away):
                self.assertEqual(len(team.starters), 9)

    def test_a_bye_player_contributes_nothing(self):
        analysis = analyze_league(
            fixture_client(), 111222, week=1, n_sims=1000, seed=1
        )
        starters = analysis.matchups[0].home.starters
        on_bye = [player for player in starters if player.nfl_team == "MIN"]
        self.assertTrue(on_bye)
        for player in on_bye:
            self.assertTrue(player.on_bye)
            self.assertEqual(player.expected, 0.0)

    def test_recommendations_are_produced(self):
        analysis = analyze_league(
            fixture_client(), 111222, week=1, n_sims=2000, seed=1
        )
        self.assertEqual(len(analysis.recommendations), 4)
        self.assertTrue(all(r.fair_american for r in analysis.recommendations))

    def test_week_one_with_no_history_still_runs(self):
        # A brand-new season has no game logs at all, which is exactly the
        # state a league is in before anyone has played.
        analysis = analyze_league(
            fixture_client(projected_only=True), 111222, week=1, n_sims=1000, seed=1
        )
        self.assertEqual(len(analysis.matchups), 2)
        for matchup in analysis.matchups:
            self.assertGreater(matchup.home.sd, 0)


if __name__ == "__main__":
    unittest.main()
