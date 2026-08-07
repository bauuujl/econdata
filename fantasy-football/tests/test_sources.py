import json
import os
import tempfile
import unittest

from ffbets.sources import (
    Ensemble,
    FileSource,
    OpponentAdjustedSource,
    RecentFormSource,
    RotowireSource,
    build_ensemble,
)


class FakeContext:
    """Stands in for a ProjectionBuilder."""

    def __init__(self, projections=None, history=None, factors=None, player_db=None):
        self.projections = projections or {}
        self.history = history or {}
        self.player_db = player_db or {}
        self._factors = factors or {}

    def defense_factor(self, team, position):
        return self._factors.get((team, position))


def context():
    return FakeContext(
        projections={
            "1": {
                "points": 18.0,
                "has_stats": True,
                "position": "WR",
                "opponent": "DAL",
                "name": "Star Receiver",
            },
            "2": {"points": 0.0, "has_stats": False, "position": "RB", "opponent": "KC"},
        },
        history={"1": [10.0, 14.0, 22.0], "2": [8.0]},
        factors={("DAL", "WR"): 1.20, ("KC", "RB"): 0.80},
    )


class RotowireTests(unittest.TestCase):
    def test_returns_the_scored_projection(self):
        self.assertEqual(RotowireSource().project("1", context()), 18.0)

    def test_declines_when_there_is_no_statline(self):
        self.assertIsNone(RotowireSource().project("2", context()))
        self.assertIsNone(RotowireSource().project("missing", context()))


class RecentFormTests(unittest.TestCase):
    def test_weights_recent_games_more_heavily(self):
        value = RecentFormSource().project("1", context())
        self.assertGreater(value, sum([10.0, 14.0, 22.0]) / 3)

    def test_declines_on_a_thin_game_log(self):
        self.assertIsNone(RecentFormSource().project("2", context()))

    def test_a_longer_half_life_moves_toward_the_plain_mean(self):
        slow = RecentFormSource(half_life=50).project("1", context())
        self.assertAlmostEqual(slow, sum([10.0, 14.0, 22.0]) / 3, delta=0.5)

    def test_a_flat_log_returns_that_value(self):
        ctx = FakeContext(history={"x": [12.0, 12.0, 12.0, 12.0]})
        self.assertAlmostEqual(RecentFormSource().project("x", ctx), 12.0)


class OpponentAdjustedTests(unittest.TestCase):
    def test_a_generous_defence_raises_the_projection(self):
        baseline = RecentFormSource().project("1", context())
        adjusted = OpponentAdjustedSource().project("1", context())
        self.assertGreater(adjusted, baseline)

    def test_the_adjustment_is_shrunk_toward_no_change(self):
        baseline = RecentFormSource().project("1", context())
        adjusted = OpponentAdjustedSource(shrinkage=0.5).project("1", context())
        # Factor 1.20 shrunk by half is 1.10, not 1.20.
        self.assertAlmostEqual(adjusted / baseline, 1.10, places=6)

    def test_declines_without_an_opponent_or_a_factor(self):
        ctx = FakeContext(
            projections={"1": {"points": 10, "has_stats": True, "position": "WR"}},
            history={"1": [10.0, 11.0, 12.0]},
        )
        self.assertIsNone(OpponentAdjustedSource().project("1", ctx))


class FileSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def write(self, name, content):
        path = os.path.join(self.directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def test_json_keyed_by_player_id(self):
        path = self.write("p.json", json.dumps({"1": 21.5}))
        self.assertEqual(FileSource(path).project("1", context()), 21.5)

    def test_csv_keyed_by_name(self):
        path = self.write("p.csv", "name,points\nStar Receiver,19.5\n")
        self.assertEqual(FileSource(path).project("1", context()), 19.5)

    def test_unknown_players_get_nothing(self):
        path = self.write("p.json", json.dumps({"999": 5.0}))
        self.assertIsNone(FileSource(path).project("1", context()))

    def test_missing_file_is_rejected(self):
        with self.assertRaises(ValueError):
            FileSource(os.path.join(self.directory, "nope.json"))


class EnsembleTests(unittest.TestCase):
    def test_weighted_mean_of_the_voters(self):
        ensemble = Ensemble([RotowireSource(0.5), RecentFormSource(0.5)])
        estimate = ensemble.combine("1", context())
        form = RecentFormSource().project("1", context())
        self.assertAlmostEqual(estimate.mean, 0.5 * 18.0 + 0.5 * form)

    def test_weights_renormalise_when_a_source_abstains(self):
        ensemble = Ensemble([RotowireSource(0.8), RecentFormSource(0.2)])
        # Player 2 has no projection and too little history, so only one votes.
        self.assertIsNone(ensemble.combine("2", context()))

        single = Ensemble([RotowireSource(0.8), RecentFormSource(0.2)])
        estimate = single.combine("1", context())
        self.assertIsNotNone(estimate)

    def test_disagreement_is_zero_when_sources_agree(self):
        ctx = FakeContext(
            projections={"x": {"points": 12.0, "has_stats": True, "position": "WR"}},
            history={"x": [12.0, 12.0, 12.0]},
        )
        estimate = Ensemble([RotowireSource(1), RecentFormSource(1)]).combine("x", ctx)
        self.assertAlmostEqual(estimate.disagreement, 0.0, places=6)

    def test_disagreement_grows_when_sources_diverge(self):
        ctx = FakeContext(
            projections={"x": {"points": 20.0, "has_stats": True, "position": "WR"}},
            history={"x": [4.0, 4.0, 4.0]},
        )
        estimate = Ensemble([RotowireSource(1), RecentFormSource(1)]).combine("x", ctx)
        self.assertGreater(estimate.disagreement, 5.0)

    def test_contributions_are_reported(self):
        estimate = Ensemble([RotowireSource(1), RecentFormSource(1)]).combine("1", context())
        self.assertIn("rotowire", estimate.contributions)
        self.assertIn("form", estimate.contributions)
        self.assertEqual(estimate.label, "proj+form")

    def test_a_broken_source_does_not_sink_the_ensemble(self):
        class Broken(RotowireSource):
            name = "broken"

            def project(self, player_id, ctx):
                raise RuntimeError("boom")

        estimate = Ensemble([Broken(1), RotowireSource(1)]).combine("1", context())
        self.assertAlmostEqual(estimate.mean, 18.0)

    def test_no_sources_means_no_estimate(self):
        self.assertIsNone(Ensemble([]).combine("1", context()))


class BuildEnsembleTests(unittest.TestCase):
    def test_parses_names_and_weights(self):
        ensemble = build_ensemble("rotowire:0.7,form:0.3")
        self.assertEqual(ensemble.names, ["rotowire", "form"])
        self.assertAlmostEqual(ensemble.sources[0].weight, 0.7)

    def test_uses_default_weights_when_omitted(self):
        ensemble = build_ensemble("rotowire,form")
        self.assertAlmostEqual(ensemble.sources[0].weight, 0.80)
        self.assertAlmostEqual(ensemble.sources[1].weight, 0.20)

    def test_rejects_unknown_sources(self):
        with self.assertRaises(ValueError):
            build_ensemble("crystalball")

    def test_file_source_needs_a_path(self):
        with self.assertRaises(ValueError):
            build_ensemble("file")

    def test_empty_spec_is_rejected(self):
        with self.assertRaises(ValueError):
            build_ensemble("")


if __name__ == "__main__":
    unittest.main()
