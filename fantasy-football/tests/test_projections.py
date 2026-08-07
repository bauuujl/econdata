import unittest

from ffbets.projections import (
    POSITION_CV,
    ProjectionBuilder,
    play_probability,
    shrunk_sd,
)

PPR = {"rec": 1.0, "rec_yd": 0.1, "rec_td": 6.0, "rush_yd": 0.1, "rush_td": 6.0}


def ewma(history, half_life=3.0):
    """Reference implementation of the form source's weighting, for tests."""
    decay = 0.5 ** (1.0 / half_life)
    weights = [decay ** age for age in range(len(history))]
    values = list(reversed(history))
    return sum(w * v for w, v in zip(weights, values)) / sum(weights)


class FakeClient:
    """Stands in for the Sleeper API with canned payloads."""

    def __init__(self, projections=None, stats=None, fail_projections=False):
        self.projections = projections or []
        self.stats = stats or {}
        self.fail_projections = fail_projections

    def week_projections(self, season, week, season_type="regular", positions=None):
        if self.fail_projections:
            raise RuntimeError("projections feed down")
        return self.projections

    def week_stats(self, season, week, season_type="regular"):
        return self.stats.get(week, {})


def projection_row(player_id, name, position, team, opponent, stats, injury=None):
    first, _, last = name.partition(" ")
    return {
        "player_id": player_id,
        "team": team,
        "opponent": opponent,
        "game_id": "game-%s" % team,
        "stats": dict(stats, gp=1.0),
        "player": {
            "first_name": first,
            "last_name": last,
            "position": position,
            "team": team,
            "injury_status": injury,
        },
    }


class PlayProbabilityTests(unittest.TestCase):
    def test_healthy_players_always_play(self):
        self.assertEqual(play_probability(None), 1.0)
        self.assertEqual(play_probability(""), 1.0)

    def test_designations_are_case_insensitive(self):
        self.assertEqual(play_probability("Questionable"), 0.75)
        self.assertEqual(play_probability("questionable"), 0.75)
        self.assertEqual(play_probability("OUT"), 0.0)
        self.assertEqual(play_probability("IR"), 0.0)

    def test_unknown_designations_do_not_bench_a_player(self):
        self.assertEqual(play_probability("Probable"), 1.0)


class ShrinkageTests(unittest.TestCase):
    def test_no_history_falls_back_to_the_positional_prior(self):
        sd, games = shrunk_sd(20.0, [], "QB")
        self.assertEqual(games, 0)
        self.assertAlmostEqual(sd, POSITION_CV["QB"] * 20.0, places=6)

    def test_a_long_steady_game_log_pulls_the_spread_down(self):
        steady = [12.0, 12.5, 11.5, 12.0, 12.2, 11.8, 12.1, 12.0, 11.9, 12.3]
        sd, games = shrunk_sd(12.0, steady, "WR")
        self.assertEqual(games, 10)
        self.assertLess(sd, POSITION_CV["WR"] * 12.0)

    def test_a_volatile_game_log_pushes_the_spread_up(self):
        boom_bust = [0.0, 34.0, 2.0, 28.0, 1.0, 31.0, 3.0, 27.0]
        sd, _ = shrunk_sd(15.0, boom_bust, "WR")
        self.assertGreater(sd, POSITION_CV["WR"] * 15.0)

    def test_more_history_moves_towards_the_observed_spread(self):
        sample = [5.0, 25.0]
        short, _ = shrunk_sd(15.0, sample, "RB")
        long, _ = shrunk_sd(15.0, sample * 12, "RB")
        observed = 10.0  # sd of the repeated 5/25 sample
        self.assertLess(abs(long - observed), abs(short - observed))

    def test_spread_never_collapses_to_zero(self):
        sd, _ = shrunk_sd(0.0, [0.0] * 12, "K")
        self.assertGreaterEqual(sd, 1.0)


class BuilderTests(unittest.TestCase):
    def build(self, **kwargs):
        rows = [
            projection_row("1", "Star Receiver", "WR", "BUF", "MIA",
                           {"rec": 7, "rec_yd": 90, "rec_td": 0.6}),
            projection_row("2", "Hurt Back", "RB", "KC", "DEN",
                           {"rush_yd": 70, "rush_td": 0.5}, injury="Questionable"),
        ]
        stats = {
            1: {"1": {"gp": 1, "rec": 6, "rec_yd": 80}, "2": {"gp": 1, "rush_yd": 60}},
            2: {"1": {"gp": 1, "rec": 8, "rec_yd": 110, "rec_td": 1}},
            3: {"1": {"gp": 1, "rec": 4, "rec_yd": 40}, "2": {"gp": 0}},
        }
        client = FakeClient(projections=rows, stats=stats, **kwargs)
        return ProjectionBuilder(client, PPR, "2025", 4, history_weeks=8)

    def test_mean_blends_projection_with_recent_form(self):
        builder = self.build()
        outlook = builder.outlook("1")
        # Projection scores 7 + 9 + 3.6 = 19.6; the game log is 14 / 25 / 8,
        # weighted toward the most recent game.
        self.assertAlmostEqual(
            outlook.mean, 0.8 * 19.6 + 0.2 * ewma([14.0, 25.0, 8.0]), places=4
        )
        self.assertEqual(outlook.source, "proj+form")
        self.assertEqual(outlook.games_used, 3)

    def test_thin_history_uses_the_projection_alone(self):
        builder = self.build()
        outlook = builder.outlook("2")
        self.assertAlmostEqual(outlook.mean, 7.0 + 3.0, places=4)
        self.assertEqual(outlook.source, "proj")

    def test_injury_status_is_carried_through(self):
        outlook = self.build().outlook("2")
        self.assertEqual(outlook.play_prob, 0.75)
        self.assertEqual(outlook.flags, "QUES")
        self.assertLess(outlook.expected, outlook.mean)

    def test_metadata_is_populated(self):
        outlook = self.build().outlook("1")
        self.assertEqual(outlook.name, "Star Receiver")
        self.assertEqual(outlook.position, "WR")
        self.assertEqual(outlook.nfl_team, "BUF")
        self.assertEqual(outlook.opponent, "MIA")

    def test_a_team_with_no_game_is_on_bye(self):
        builder = self.build()
        builder.player_db["99"] = {
            "first_name": "Bye",
            "last_name": "Week",
            "position": "TE",
            "team": "CLE",
        }
        outlook = builder.outlook("99")
        self.assertTrue(outlook.on_bye)
        self.assertEqual(outlook.play_prob, 0.0)
        self.assertEqual(outlook.expected, 0.0)
        self.assertEqual(outlook.flags, "BYE")

    def test_unknown_players_get_replacement_level(self):
        outlook = self.build().outlook("nobody")
        self.assertEqual(outlook.source, "replacement")
        self.assertGreater(outlook.mean, 0.0)

    def test_the_model_survives_a_missing_projections_feed(self):
        builder = self.build(fail_projections=True)
        self.assertEqual(builder.projections, {})
        outlook = builder.outlook("1")
        self.assertEqual(outlook.source, "form")
        self.assertAlmostEqual(outlook.mean, ewma([14.0, 25.0, 8.0]), places=4)

    def test_weeks_a_player_missed_are_left_out_of_the_game_log(self):
        # Player 2 has gp=0 in week 3, so only week 1 counts.
        self.assertEqual(self.build().history["2"], [6.0])

    def test_outlooks_skips_empty_slots(self):
        outlooks = self.build().outlooks(["1", "0", "", None, "2"])
        self.assertEqual([o.player_id for o in outlooks], ["1", "2"])


if __name__ == "__main__":
    unittest.main()
