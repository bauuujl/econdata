import unittest

from ffbets.analyzer import (
    build_recommendations,
    match_team,
    pair_matchups,
    team_label,
)
from ffbets.simulate import TeamResult, compare


def team(roster_id, name, owner="owner", totals=None):
    result = TeamResult(roster_id=roster_id, name=name, owner=owner)
    result.totals = totals or []
    return result


def matchup(p_home, name_home="Home", name_away="Away", matchup_id=1):
    """A matchup with an exact win split, built from synthetic totals."""
    count = 1000
    home_wins = int(round(p_home * count))
    # Roster IDs must be unique across matchups, as they are in a real league.
    home = team(matchup_id * 2 - 1, name_home, totals=[1.0] * count)
    away = team(
        matchup_id * 2, name_away, totals=[0.0] * home_wins + [2.0] * (count - home_wins)
    )
    return compare(matchup_id, home, away)


class LabelTests(unittest.TestCase):
    def test_prefers_the_custom_team_name(self):
        user = {"display_name": "blake", "metadata": {"team_name": "LamaRPG"}}
        self.assertEqual(team_label(user, 1), "LamaRPG")

    def test_falls_back_to_the_handle_then_the_roster(self):
        self.assertEqual(team_label({"display_name": "blake"}, 1), "blake")
        self.assertEqual(team_label({"metadata": None}, 4), "Roster 4")
        self.assertEqual(team_label(None, 7), "Roster 7")


class PairingTests(unittest.TestCase):
    def test_groups_rosters_by_matchup(self):
        feed = [
            {"roster_id": 1, "matchup_id": 1},
            {"roster_id": 2, "matchup_id": 1},
            {"roster_id": 3, "matchup_id": 2},
        ]
        self.assertEqual(pair_matchups(feed), {1: [1, 2], 2: [3]})


class MatchTeamTests(unittest.TestCase):
    def setUp(self):
        self.teams = [team(1, "Sunday Scaries", "blake"), team(2, "Pylon Patrol", "jose")]

    def test_matches_on_roster_id_name_and_handle(self):
        self.assertIs(match_team("1", self.teams), self.teams[0])
        self.assertIs(match_team("Pylon Patrol", self.teams), self.teams[1])
        self.assertIs(match_team("BLAKE", self.teams), self.teams[0])

    def test_matches_on_a_unique_fragment(self):
        self.assertIs(match_team("pylon", self.teams), self.teams[1])

    def test_ambiguous_and_unknown_keys_return_nothing(self):
        crowd = self.teams + [team(3, "Pylon Pushers", "sam")]
        self.assertIsNone(match_team("pylon", crowd))
        self.assertIsNone(match_team("nobody", self.teams))


class RecommendationTests(unittest.TestCase):
    def test_unpriced_recommendations_rank_by_win_probability(self):
        results = [matchup(0.70, "A", "B")]
        recommendations = build_recommendations(results, None, 1000, 0.25)
        self.assertEqual([r.team for r in recommendations], ["A", "B"])
        self.assertFalse(any(r.priced for r in recommendations))
        self.assertAlmostEqual(recommendations[0].model_probability, 0.70)

    def test_fair_prices_invert_the_model_probability(self):
        from ffbets.odds import implied_probability

        recommendation = build_recommendations([matchup(0.75, "A", "B")], None, 1000, 0.25)[0]
        self.assertAlmostEqual(
            implied_probability(recommendation.fair_american), 0.75, places=2
        )

    def test_a_generous_price_produces_a_positive_edge_and_a_stake(self):
        results = [matchup(0.60, "A", "B")]
        # +150 pays like a 40% shot; the model says 60%.
        priced = build_recommendations(results, {"A": "+150", "B": "-200"}, 1000, 0.25)
        best = priced[0]
        self.assertEqual(best.team, "A")
        self.assertGreater(best.expected_value, 0)
        self.assertGreater(best.edge, 0)
        self.assertGreater(best.kelly, 0)
        self.assertIsNotNone(best.devig_probability)
        self.assertGreater(best.market_hold, 0)

    def test_a_stingy_price_is_a_no_bet(self):
        priced = build_recommendations([matchup(0.55, "A", "B")], {"A": "-200"}, 1000, 0.25)
        side = next(r for r in priced if r.team == "A")
        self.assertLess(side.expected_value, 0)
        self.assertEqual(side.kelly, 0.0)

    def test_recommendations_sort_by_expected_value(self):
        results = [matchup(0.60, "A", "B"), matchup(0.60, "C", "D", matchup_id=2)]
        priced = build_recommendations(
            results, {"A": "+150", "B": "-200", "C": "-300", "D": "+250"}, 1000, 0.25
        )
        values = [r.expected_value for r in priced if r.priced]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual(priced[0].team, "A")

    def test_one_sided_prices_fall_back_to_the_raw_implied_probability(self):
        priced = build_recommendations([matchup(0.60, "A", "B")], {"A": "+150"}, 1000, 0.25)
        side = next(r for r in priced if r.team == "A")
        self.assertIsNone(side.devig_probability)
        self.assertAlmostEqual(side.implied_probability, 0.40)

    def test_unknown_team_names_in_the_odds_file_are_ignored(self):
        priced = build_recommendations([matchup(0.60, "A", "B")], {"Nobody": "+150"}, 1000, 0.25)
        self.assertFalse(any(r.priced for r in priced))


if __name__ == "__main__":
    unittest.main()
