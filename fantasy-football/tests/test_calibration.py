import random
import unittest

from ffbets import calibration as calib


def overconfident_sample(count=600, stretch=1.6, seed=0):
    """A forecaster that ranks well but states probabilities too extreme."""
    generator = random.Random(seed)
    truth = [generator.uniform(0.25, 0.9) for _ in range(count)]
    stated = [
        min(max(0.5 + (p - 0.5) * stretch, 0.02), 0.98) for p in truth
    ]
    outcomes = [1 if generator.random() < p else 0 for p in truth]
    weeks = [index // 10 for index in range(count)]
    return stated, outcomes, weeks


class ScoringRuleTests(unittest.TestCase):
    def test_brier_of_a_perfect_forecaster_is_zero(self):
        self.assertAlmostEqual(calib.brier_score([1.0, 0.0], [1, 0]), 0.0)

    def test_brier_of_a_coin_flip(self):
        self.assertAlmostEqual(calib.brier_score([0.5] * 4, [1, 0, 1, 0]), 0.25)

    def test_log_loss_of_a_coin_flip(self):
        self.assertAlmostEqual(calib.log_loss([0.5] * 4, [1, 0, 1, 0]), 0.6931, places=4)

    def test_log_loss_is_finite_even_when_confidently_wrong(self):
        self.assertLess(calib.log_loss([1.0], [0]), 1e3)

    def test_empty_input_is_safe(self):
        self.assertEqual(calib.brier_score([], []), 0.0)
        self.assertEqual(calib.log_loss([], []), 0.0)


class MurphyTests(unittest.TestCase):
    def test_the_identity_holds_exactly(self):
        stated, outcomes, _ = overconfident_sample()
        parts = calib.murphy_decomposition(stated, outcomes, bins=6)
        self.assertAlmostEqual(
            parts.binned_brier,
            parts.reliability - parts.resolution + parts.uncertainty,
            places=12,
        )

    def test_binning_error_is_small(self):
        stated, outcomes, _ = overconfident_sample()
        parts = calib.murphy_decomposition(stated, outcomes, bins=6)
        self.assertLess(abs(parts.discretization_gap), 0.01)

    def test_a_calibrated_forecaster_has_almost_no_reliability_term(self):
        generator = random.Random(3)
        truth = [generator.uniform(0.2, 0.9) for _ in range(2000)]
        outcomes = [1 if generator.random() < p else 0 for p in truth]
        parts = calib.murphy_decomposition(truth, outcomes, bins=6)
        self.assertLess(parts.reliability, 0.005)

    def test_an_uninformative_forecaster_has_no_resolution(self):
        outcomes = [1, 0] * 200
        parts = calib.murphy_decomposition([0.5] * 400, outcomes, bins=4)
        self.assertAlmostEqual(parts.resolution, 0.0, places=6)
        self.assertAlmostEqual(parts.uncertainty, 0.25, places=6)

    def test_overconfidence_shows_up_as_reliability(self):
        stated, outcomes, _ = overconfident_sample()
        parts = calib.murphy_decomposition(stated, outcomes, bins=6)
        self.assertGreater(parts.reliability, 0.002)

    def test_uncertainty_matches_the_base_rate(self):
        outcomes = [1] * 30 + [0] * 70
        parts = calib.murphy_decomposition([0.5] * 100, outcomes, bins=3)
        self.assertAlmostEqual(parts.uncertainty, 0.3 * 0.7, places=9)


class ReliabilityCurveTests(unittest.TestCase):
    def test_bins_cover_every_forecast(self):
        stated, outcomes, _ = overconfident_sample(count=200)
        curve = calib.reliability_curve(stated, outcomes, bins=5)
        self.assertEqual(sum(item.count for item in curve), 200)

    def test_quantile_bins_are_roughly_balanced(self):
        stated, outcomes, _ = overconfident_sample(count=200)
        counts = [item.count for item in calib.reliability_curve(stated, outcomes, 5)]
        self.assertLess(max(counts) - min(counts), 20)

    def test_intervals_bracket_the_observed_rate(self):
        stated, outcomes, _ = overconfident_sample(count=200)
        for item in calib.reliability_curve(stated, outcomes, bins=5):
            self.assertLessEqual(item.interval_low, item.actual)
            self.assertGreaterEqual(item.interval_high, item.actual)

    def test_intervals_narrow_as_the_sample_grows(self):
        narrow = calib.wilson_interval(50, 100)
        wide = calib.wilson_interval(5, 10)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])


class PlattTests(unittest.TestCase):
    def test_detects_overconfidence_as_a_slope_below_one(self):
        stated, outcomes, _ = overconfident_sample()
        model = calib.PlattCalibrator().fit(stated, outcomes)
        self.assertLess(model.slope, 1.0)
        self.assertIn("overconfident", model.description)

    def test_pulls_extreme_forecasts_toward_the_middle(self):
        stated, outcomes, _ = overconfident_sample()
        model = calib.PlattCalibrator().fit(stated, outcomes)
        self.assertLess(model.predict(0.95), 0.95)
        self.assertGreater(model.predict(0.05), 0.05)

    def test_stays_monotone(self):
        stated, outcomes, _ = overconfident_sample()
        model = calib.PlattCalibrator().fit(stated, outcomes)
        values = [model.predict(x / 20) for x in range(1, 20)]
        self.assertEqual(values, sorted(values))

    def test_survives_a_perfectly_separable_sample(self):
        model = calib.PlattCalibrator().fit([0.2, 0.3, 0.7, 0.8], [0, 0, 1, 1])
        self.assertTrue(0.0 < model.predict(0.5) < 1.0)

    def test_round_trips_through_a_dict(self):
        stated, outcomes, _ = overconfident_sample(count=200)
        model = calib.PlattCalibrator().fit(stated, outcomes)
        restored = calib.calibrator_from_dict(model.to_dict())
        self.assertAlmostEqual(restored.predict(0.7), model.predict(0.7), places=9)


class IsotonicTests(unittest.TestCase):
    def test_output_is_monotone(self):
        stated, outcomes, _ = overconfident_sample()
        model = calib.IsotonicCalibrator().fit(stated, outcomes)
        values = [model.predict(x / 40) for x in range(1, 40)]
        self.assertEqual(values, sorted(values))

    def test_recovers_a_step_relationship(self):
        stated = [0.1] * 50 + [0.9] * 50
        outcomes = [0] * 50 + [1] * 50
        model = calib.IsotonicCalibrator().fit(stated, outcomes)
        self.assertLess(model.predict(0.1), 0.2)
        self.assertGreater(model.predict(0.9), 0.8)

    def test_clips_outside_the_training_range(self):
        model = calib.IsotonicCalibrator().fit([0.3, 0.4, 0.6, 0.7], [0, 0, 1, 1])
        self.assertEqual(model.predict(0.01), model.predict(0.3))
        self.assertEqual(model.predict(0.99), model.predict(0.7))

    def test_round_trips_through_a_dict(self):
        stated, outcomes, _ = overconfident_sample(count=200)
        model = calib.IsotonicCalibrator().fit(stated, outcomes)
        restored = calib.calibrator_from_dict(model.to_dict())
        self.assertAlmostEqual(restored.predict(0.6), model.predict(0.6), places=9)


class CrossValidationTests(unittest.TestCase):
    def test_folds_never_split_a_group(self):
        groups = [index // 5 for index in range(50)]
        folds = calib.grouped_folds(groups, folds=5, seed=1)
        self.assertEqual(sum(len(fold) for fold in folds), 50)
        for fold in folds:
            weeks_in_fold = {groups[index] for index in fold}
            for other in folds:
                if other is fold:
                    continue
                self.assertFalse(weeks_in_fold & {groups[i] for i in other})

    def test_platt_helps_a_genuinely_overconfident_forecaster(self):
        stated, outcomes, weeks = overconfident_sample()
        report = calib.cross_validated_calibration(
            stated, outcomes, weeks, method="platt", folds=5, seed=1
        )
        self.assertGreater(report.improvement, 0)

    def test_every_forecast_is_calibrated_out_of_sample(self):
        stated, outcomes, weeks = overconfident_sample(count=200)
        report = calib.cross_validated_calibration(stated, outcomes, weeks, "platt", 5)
        self.assertEqual(len(report.calibrated), 200)
        self.assertTrue(all(value is not None for value in report.calibrated))

    def test_a_deployable_map_is_fitted_on_everything(self):
        stated, outcomes, weeks = overconfident_sample(count=200)
        report = calib.cross_validated_calibration(stated, outcomes, weeks, "platt", 5)
        self.assertIsNotNone(report.fitted)
        self.assertLess(report.fitted.slope, 1.0)

    def test_a_one_sided_sample_does_not_crash(self):
        report = calib.cross_validated_calibration(
            [0.4, 0.5, 0.6, 0.7], [1, 1, 1, 1], [1, 2, 3, 4], "platt", 2
        )
        self.assertEqual(len(report.calibrated), 4)

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            calib.make_calibrator("magic")


if __name__ == "__main__":
    unittest.main()
