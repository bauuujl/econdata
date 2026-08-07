import statistics
import unittest

from ffbets.projections import PlayerOutlook
from ffbets.simulate import (
    CorrelationModel,
    WeekSimulator,
    _lognormal_parameters,
    _percentile,
    compare,
    standard_error,
    summarize_team,
)


def player(player_id, mean, sd, position="WR", team="BUF", opponent="MIA", play_prob=1.0):
    return PlayerOutlook(
        player_id=player_id,
        name=player_id,
        position=position,
        nfl_team=team,
        opponent=opponent,
        mean=mean,
        sd=sd,
        play_prob=play_prob,
    )


class DistributionTests(unittest.TestCase):
    def test_lognormal_parameters_recover_mean_and_sd(self):
        import math

        mu, sigma = _lognormal_parameters(12.0, 6.0)
        mean = math.exp(mu + sigma * sigma / 2)
        variance = (math.exp(sigma * sigma) - 1) * math.exp(2 * mu + sigma * sigma)
        self.assertAlmostEqual(mean, 12.0, places=6)
        self.assertAlmostEqual(math.sqrt(variance), 6.0, places=6)

    def test_percentiles(self):
        values = list(range(101))
        self.assertAlmostEqual(_percentile(values, 0.0), 0)
        self.assertAlmostEqual(_percentile(values, 0.5), 50)
        self.assertAlmostEqual(_percentile(values, 1.0), 100)
        self.assertEqual(_percentile([], 0.5), 0.0)


class SimulationTests(unittest.TestCase):
    def test_team_total_matches_the_sum_of_player_means(self):
        starters = [player("p%d" % i, 12.0, 5.0, team="T%d" % i, opponent="X%d" % i)
                    for i in range(9)]
        simulator = WeekSimulator(n_sims=6000, seed=3)
        totals = simulator.run([(1, starters)])[1]
        self.assertAlmostEqual(statistics.mean(totals), 9 * 12.0, delta=1.0)

    def test_scores_are_never_negative(self):
        starters = [player("p", 2.0, 8.0)]
        totals = WeekSimulator(n_sims=2000, seed=4).run([(1, starters)])[1]
        self.assertGreaterEqual(min(totals), 0.0)

    def test_players_who_cannot_play_contribute_nothing(self):
        starters = [player("out", 20.0, 8.0, play_prob=0.0)]
        totals = WeekSimulator(n_sims=500, seed=5).run([(1, starters)])[1]
        self.assertEqual(max(totals), 0.0)

    def test_play_probability_scales_the_mean(self):
        starters = [player("iffy", 20.0, 6.0, play_prob=0.5)]
        totals = WeekSimulator(n_sims=8000, seed=6).run([(1, starters)])[1]
        self.assertAlmostEqual(statistics.mean(totals), 10.0, delta=0.8)

    def test_stacking_teammates_widens_the_outcome(self):
        # Same expected points either way; the stacked roster should swing
        # harder because its players share an offence.
        stacked = [player("s%d" % i, 12.0, 6.0, team="BUF", opponent="MIA")
                   for i in range(6)]
        spread = [player("d%d" % i, 12.0, 6.0, team="T%d" % i, opponent="O%d" % i)
                  for i in range(6)]
        simulator = WeekSimulator(n_sims=8000, seed=7)
        results = simulator.run([(1, stacked), (2, spread)])
        self.assertGreater(statistics.pstdev(results[1]), statistics.pstdev(results[2]))

    def test_zero_correlation_is_honoured(self):
        stacked = [player("s%d" % i, 12.0, 6.0) for i in range(6)]
        loose = WeekSimulator(
            correlation=CorrelationModel(offense=0.0, kicker=0.0, defense=0.0),
            n_sims=8000,
            seed=8,
        )
        tight = WeekSimulator(correlation=CorrelationModel(offense=0.7), n_sims=8000, seed=8)
        self.assertLess(
            statistics.pstdev(loose.run([(1, stacked)])[1]),
            statistics.pstdev(tight.run([(1, stacked)])[1]),
        )

    def test_same_seed_reproduces_the_run(self):
        starters = [player("p", 12.0, 5.0)]
        first = WeekSimulator(n_sims=200, seed=99).run([(1, starters)])[1]
        second = WeekSimulator(n_sims=200, seed=99).run([(1, starters)])[1]
        self.assertEqual(first, second)


class ComparisonTests(unittest.TestCase):
    def _matchup(self, home_mean, away_mean, n_sims=8000, seed=11):
        home = [player("h%d" % i, home_mean / 5, 4.0, team="H%d" % i, opponent="A%d" % i)
                for i in range(5)]
        away = [player("a%d" % i, away_mean / 5, 4.0, team="B%d" % i, opponent="C%d" % i)
                for i in range(5)]
        totals = WeekSimulator(n_sims=n_sims, seed=seed).run([(1, home), (2, away)])
        return compare(
            1,
            summarize_team(1, "Home", "h", home, totals[1]),
            summarize_team(2, "Away", "a", away, totals[2]),
        )

    def test_probabilities_sum_to_one(self):
        result = self._matchup(120, 100)
        self.assertAlmostEqual(result.p_home + result.p_away + result.p_tie, 1.0, places=6)

    def test_the_better_team_is_favoured(self):
        result = self._matchup(130, 100)
        self.assertGreater(result.p_home, 0.5)
        self.assertIs(result.favorite, result.home)
        self.assertIs(result.underdog, result.away)

    def test_even_teams_are_a_coin_flip(self):
        result = self._matchup(110, 110)
        self.assertAlmostEqual(result.p_home, 0.5, delta=0.03)
        self.assertLess(result.confidence, 0.08)

    def test_margin_summary_reads_from_the_favourites_side(self):
        result = self._matchup(100, 130)  # away is the favourite
        median, low, high = result.favorite_margin
        self.assertGreater(median, 0)
        self.assertLess(low, high)

    def test_summary_statistics_are_ordered(self):
        result = self._matchup(120, 100)
        self.assertLess(result.home.p10, result.home.p50)
        self.assertLess(result.home.p50, result.home.p90)


class ErrorTests(unittest.TestCase):
    def test_standard_error_shrinks_with_more_simulations(self):
        self.assertGreater(standard_error(0.5, 1000), standard_error(0.5, 100000))
        self.assertAlmostEqual(standard_error(0.5, 10000), 0.005, places=4)

    def test_certainty_has_no_error(self):
        self.assertEqual(standard_error(1.0, 1000), 0.0)
        self.assertEqual(standard_error(0.5, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
