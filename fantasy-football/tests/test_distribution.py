"""Tests for the distributional scores and the simulator-level calibration."""

import math
import random
import statistics
import unittest

from ffbets.calibration import (
    ScoreCalibration,
    crps_from_samples,
    cross_validated_score_calibration,
    fit_score_calibration,
    pit_summary,
    pit_value,
    score_calibration_from_dict,
)
from ffbets.projections import PlayerOutlook
from ffbets.simulate import CorrelationModel, WeekSimulator


def normal_samples(mean=0.0, sd=1.0, count=20000, seed=0):
    generator = random.Random(seed)
    return sorted(generator.gauss(mean, sd) for _ in range(count))


class CrpsTests(unittest.TestCase):
    def test_a_point_forecast_at_the_truth_scores_zero(self):
        self.assertAlmostEqual(crps_from_samples([5.0] * 200, 5.0), 0.0, places=9)

    def test_a_point_forecast_scores_its_absolute_error(self):
        self.assertAlmostEqual(crps_from_samples([5.0] * 200, 8.0), 3.0, places=9)

    def test_matches_the_analytic_value_for_a_standard_normal(self):
        # For N(mu, sigma) at y, CRPS is
        #   sigma * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],  z = (y-mu)/sigma
        # which at z = 0 collapses to 2*phi(0) - 1/sqrt(pi) = 0.23369.
        analytic = math.sqrt(2 / math.pi) - 1 / math.sqrt(math.pi)
        self.assertAlmostEqual(analytic, 0.23369, places=5)
        self.assertAlmostEqual(crps_from_samples(normal_samples(), 0.0), analytic, places=2)

    def test_scales_with_the_units_of_the_outcome(self):
        tight = crps_from_samples(normal_samples(sd=1.0), 0.0)
        wide = crps_from_samples(normal_samples(sd=2.0), 0.0)
        self.assertAlmostEqual(wide / tight, 2.0, places=1)

    def test_a_sharper_correct_forecast_beats_a_vaguer_one(self):
        self.assertLess(
            crps_from_samples(normal_samples(sd=1.0), 0.0),
            crps_from_samples(normal_samples(sd=5.0, seed=1), 0.0),
        )

    def test_a_confidently_wrong_forecast_is_punished(self):
        self.assertGreater(
            crps_from_samples(normal_samples(sd=1.0), 6.0),
            crps_from_samples(normal_samples(sd=5.0, seed=1), 6.0),
        )

    def test_empty_sample(self):
        self.assertEqual(crps_from_samples([], 3.0), 0.0)


class PitTests(unittest.TestCase):
    def test_the_median_lands_in_the_middle(self):
        self.assertAlmostEqual(pit_value(normal_samples(), 0.0), 0.5, delta=0.02)

    def test_the_tails_land_at_the_ends(self):
        samples = normal_samples()
        self.assertLess(pit_value(samples, -4.0), 0.01)
        self.assertGreater(pit_value(samples, 4.0), 0.99)

    def test_a_calibrated_forecaster_gives_uniform_values(self):
        generator = random.Random(4)
        values = []
        for _ in range(3000):
            truth = generator.gauss(0, 10)
            samples = sorted(generator.gauss(truth, 5) for _ in range(200))
            values.append(pit_value(samples, generator.gauss(truth, 5)))
        summary = pit_summary(values)
        self.assertAlmostEqual(summary.mean, 0.5, delta=0.03)
        self.assertAlmostEqual(summary.dispersion_ratio, 1.0, delta=0.06)
        self.assertIn("about right", summary.verdict)

    def test_too_narrow_a_forecast_shows_up_as_high_dispersion(self):
        generator = random.Random(5)
        values = []
        for _ in range(3000):
            samples = sorted(generator.gauss(0, 2) for _ in range(200))
            values.append(pit_value(samples, generator.gauss(0, 6)))
        summary = pit_summary(values)
        self.assertGreater(summary.dispersion_ratio, 1.1)
        self.assertIn("overconfident", summary.verdict)

    def test_too_wide_a_forecast_shows_up_as_low_dispersion(self):
        generator = random.Random(6)
        values = []
        for _ in range(3000):
            samples = sorted(generator.gauss(0, 12) for _ in range(200))
            values.append(pit_value(samples, generator.gauss(0, 3)))
        summary = pit_summary(values)
        self.assertLess(summary.dispersion_ratio, 0.9)
        self.assertIn("underconfident", summary.verdict)

    def test_coverage_is_reported(self):
        summary = pit_summary([index / 1000 for index in range(1000)])
        self.assertAlmostEqual(summary.coverage_50, 0.5, delta=0.02)
        self.assertAlmostEqual(summary.coverage_80, 0.8, delta=0.02)


class ScoreCalibrationTests(unittest.TestCase):
    def test_identity_leaves_a_player_alone(self):
        calibration = ScoreCalibration()
        self.assertTrue(calibration.is_identity)
        self.assertEqual(calibration.apply(12.0, 6.0), (12.0, 6.0))

    def test_mean_scale_moves_both_mean_and_spread(self):
        # Scaling a player's mean without their spread would silently change
        # their coefficient of variation.
        mean, sd = ScoreCalibration(mean_scale=1.1).apply(10.0, 5.0)
        self.assertAlmostEqual(mean, 11.0)
        self.assertAlmostEqual(sd, 5.5)

    def test_dispersion_moves_only_the_spread(self):
        mean, sd = ScoreCalibration(dispersion=1.2).apply(10.0, 5.0)
        self.assertAlmostEqual(mean, 10.0)
        self.assertAlmostEqual(sd, 6.0)

    def test_recovers_a_known_bias_and_dispersion(self):
        generator = random.Random(7)
        rows = []
        for _ in range(400):
            projected = generator.uniform(90, 150)
            simulated_sd = 25.0
            actual = projected * 1.08 + generator.gauss(0, 25.0 * 1.08 * 1.15)
            rows.append((projected, simulated_sd, actual))
        fitted = fit_score_calibration(rows)
        self.assertAlmostEqual(fitted.mean_scale, 1.08, delta=0.02)
        self.assertAlmostEqual(fitted.dispersion, 1.15, delta=0.08)

    def test_a_perfect_model_fits_the_identity(self):
        generator = random.Random(8)
        rows = [
            (120.0, 25.0, 120.0 + generator.gauss(0, 25.0)) for _ in range(400)
        ]
        fitted = fit_score_calibration(rows)
        self.assertAlmostEqual(fitted.mean_scale, 1.0, delta=0.02)
        self.assertAlmostEqual(fitted.dispersion, 1.0, delta=0.08)

    def test_a_tiny_sample_declines_to_fit(self):
        self.assertTrue(fit_score_calibration([(120, 25, 130)] * 3).is_identity)

    def test_corrections_are_bounded(self):
        rows = [(10.0, 1.0, 500.0)] * 50
        fitted = fit_score_calibration(rows)
        self.assertLessEqual(fitted.mean_scale, 2.0)
        self.assertLessEqual(fitted.dispersion, 2.0)

    def test_rescaling_margins_matches_applying_it_to_players(self):
        samples = [-20.0, -5.0, 5.0, 30.0]
        calibration = ScoreCalibration(mean_scale=1.1, dispersion=1.2)
        rescaled = calibration.rescale_margin(samples)
        centre = statistics.mean(samples)
        self.assertAlmostEqual(statistics.mean(rescaled), centre * 1.1, places=6)
        self.assertAlmostEqual(
            statistics.pstdev(rescaled) / statistics.pstdev(samples), 1.1 * 1.2, places=6
        )

    def test_round_trips_through_a_dict(self):
        calibration = ScoreCalibration(mean_scale=1.05, dispersion=1.12)
        restored = score_calibration_from_dict(calibration.to_dict())
        self.assertAlmostEqual(restored.mean_scale, 1.05)
        self.assertAlmostEqual(restored.dispersion, 1.12)

    def test_cross_validation_produces_a_probability_for_every_row(self):
        generator = random.Random(9)
        rows = []
        groups = []
        for index in range(120):
            samples = sorted(generator.gauss(5, 40) for _ in range(300))
            rows.append((120.0, 40.0, 125.0, samples, generator.random() < 0.5))
            groups.append(index // 10)
        fitted, probabilities = cross_validated_score_calibration(rows, groups, folds=4)
        self.assertEqual(len(probabilities), 120)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probabilities))
        self.assertIsInstance(fitted, ScoreCalibration)

    def test_empty_input(self):
        fitted, probabilities = cross_validated_score_calibration([], [])
        self.assertTrue(fitted.is_identity)
        self.assertEqual(probabilities, [])


def outlook(pid, team, opponent):
    return PlayerOutlook(
        player_id=pid,
        name=pid,
        position="WR",
        nfl_team=team,
        opponent=opponent,
        mean=13.0,
        sd=7.0,
        play_prob=1.0,
    )


class SlateFactorTests(unittest.TestCase):
    """The week-wide shock must inflate team totals without touching margins."""

    def rosters(self):
        home = [outlook("h%d" % i, "T%d" % i, "O%d" % i) for i in range(9)]
        away = [outlook("a%d" % i, "U%d" % i, "V%d" % i) for i in range(9)]
        return [(1, home), (2, away)]

    def run_with(self, slate):
        totals = WeekSimulator(
            correlation=CorrelationModel(slate_share=slate), n_sims=12000, seed=4
        ).run(self.rosters())
        margins = [h - a for h, a in zip(totals[1], totals[2])]
        return statistics.pstdev(totals[1]), statistics.pstdev(margins), totals

    def test_a_slate_shock_widens_team_totals(self):
        without, _, _ = self.run_with(0.0)
        with_slate, _, _ = self.run_with(0.20)
        self.assertGreater(with_slate, without * 1.03)

    def test_it_leaves_the_margin_alone(self):
        _, margin_without, _ = self.run_with(0.0)
        _, margin_with, _ = self.run_with(0.20)
        # The shock lands on both sides, so it cancels in the difference.
        self.assertAlmostEqual(margin_with / margin_without, 1.0, delta=0.03)

    def test_it_correlates_opposing_teams(self):
        _, _, without = self.run_with(0.0)
        _, _, with_slate = self.run_with(0.20)
        self.assertLess(abs(statistics.correlation(without[1], without[2])), 0.05)
        self.assertGreater(statistics.correlation(with_slate[1], with_slate[2]), 0.08)

    def test_the_mean_is_untouched(self):
        without, _, totals_without = self.run_with(0.0)
        _, _, totals_with = self.run_with(0.20)
        self.assertAlmostEqual(
            statistics.mean(totals_with[1]), statistics.mean(totals_without[1]), delta=1.5
        )

    def test_variance_weights_still_sum_to_one(self):
        # Adding a third level must not quietly inflate everything.
        model = CorrelationModel(slate_share=0.2, game_share=0.35)
        slate = model.slate_share
        total = slate + (1 - slate) * model.game_share + (1 - slate) * (1 - model.game_share)
        self.assertAlmostEqual(total, 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
