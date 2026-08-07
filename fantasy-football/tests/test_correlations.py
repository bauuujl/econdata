import math
import random
import unittest

from ffbets.correlations import (
    CorrelationFit,
    ObservedCorrelation,
    fit_loadings,
    latent_correlation,
    pearson,
    sigma_log,
    stack_intensity,
    tail_comparison,
)
from ffbets.projections import PlayerOutlook
from ffbets.simulate import CorrelationModel


def observation(kind, role_a, role_b, latent, count=200):
    return ObservedCorrelation(
        kind=kind, role_a=role_a, role_b=role_b, observed=latent, latent=latent, count=count
    )


class PearsonTests(unittest.TestCase):
    def test_perfect_correlation(self):
        self.assertAlmostEqual(pearson([(1, 2), (2, 4), (3, 6)]), 1.0)

    def test_perfect_anticorrelation(self):
        self.assertAlmostEqual(pearson([(1, 6), (2, 4), (3, 2)]), -1.0)

    def test_no_relationship(self):
        # Symmetric about the mean of x, so the cross-products cancel exactly.
        self.assertAlmostEqual(pearson([(1, 1), (2, 2), (3, 2), (4, 1)]), 0.0, places=9)

    def test_degenerate_input(self):
        self.assertIsNone(pearson([(1, 1)]))
        self.assertIsNone(pearson([(1, 5), (1, 6), (1, 7)]))


class LatentCorrelationTests(unittest.TestCase):
    def test_zero_maps_to_zero(self):
        self.assertAlmostEqual(latent_correlation(0.0, "QB", "WR"), 0.0, places=9)

    def test_the_lognormal_transform_compresses_so_latent_exceeds_observed(self):
        self.assertGreater(latent_correlation(0.35, "WR", "WR"), 0.35)

    def test_noisier_positions_compress_more(self):
        # A pair of high-variance positions loses more correlation through the
        # transform, so recovering it needs a bigger latent value.
        quiet = latent_correlation(0.30, "QB", "QB")
        noisy = latent_correlation(0.30, "DEF", "DEF")
        self.assertGreater(noisy, quiet)

    def test_it_inverts_the_forward_formula(self):
        for rho in (0.2, 0.45, 0.7):
            sx = sigma_log("WR")
            scale = math.sqrt((math.exp(sx * sx) - 1.0) ** 2)
            observed = (math.exp(rho * sx * sx) - 1.0) / scale
            self.assertAlmostEqual(latent_correlation(observed, "WR", "WR"), rho, places=6)

    def test_stays_inside_valid_bounds(self):
        self.assertLessEqual(latent_correlation(0.99, "WR", "WR"), 0.99)
        self.assertGreaterEqual(latent_correlation(-0.99, "WR", "WR"), -0.99)


class FitLoadingsTests(unittest.TestCase):
    def test_recovers_loadings_from_consistent_observations(self):
        truth = {"QB": 0.70, "WR": 0.50, "TE": 0.30, "RB": 0.20, "K": 0.10}
        game_share, defense = 0.40, 0.55
        observations = [
            observation("same_team", "QB", "WR1", truth["QB"] * truth["WR"]),
            observation("same_team", "QB", "TE1", truth["QB"] * truth["TE"]),
            observation("same_team", "QB", "RB1", truth["QB"] * truth["RB"]),
            observation("same_team", "QB", "K", truth["QB"] * truth["K"]),
            observation("same_team", "WR1", "TE1", truth["WR"] * truth["TE"]),
            observation("opposing", "QB", "QB", game_share * truth["QB"] ** 2),
            observation("opposing", "QB", "WR1", game_share * truth["QB"] * truth["WR"]),
            observation("defense", "DEF", "QB", -defense * truth["QB"]),
            observation("defense", "DEF", "WR1", -defense * truth["WR"]),
        ]
        loadings, fitted_share, fitted_defense, residual = fit_loadings(observations)
        for position, value in truth.items():
            self.assertAlmostEqual(loadings[position], value, places=2)
        self.assertAlmostEqual(fitted_share, game_share, places=2)
        self.assertAlmostEqual(fitted_defense, defense, places=2)
        self.assertLess(residual, 0.01)

    def test_loadings_stay_in_range(self):
        observations = [observation("same_team", "QB", "WR1", 0.99)] * 3
        loadings, share, defense, _ = fit_loadings(observations)
        for value in loadings.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 0.95)
        self.assertLessEqual(share, 0.95)
        self.assertLessEqual(defense, 0.95)

    def test_a_negative_teammate_pair_shows_up_as_misfit(self):
        # A one-factor model cannot make teammates anticorrelate, so this must
        # surface as a residual rather than being absorbed.
        observations = [
            observation("same_team", "QB", "WR1", 0.40),
            observation("same_team", "WR1", "WR2", -0.05),
        ]
        loadings, share, defense, _ = fit_loadings(observations)
        fit = CorrelationFit(
            loadings=loadings, game_share=share, defense=defense, observations=observations
        )
        worst = fit.residuals()[0]
        self.assertEqual(worst["pair"], "WR1-WR2")
        self.assertLess(worst["residual"], -0.1)

    def test_no_observations_leaves_the_defaults(self):
        loadings, share, defense, residual = fit_loadings([])
        self.assertAlmostEqual(share, 0.35)
        self.assertAlmostEqual(defense, 0.30)
        self.assertEqual(residual, 0.0)


class FitObjectTests(unittest.TestCase):
    def build(self):
        return CorrelationFit(
            loadings={"QB": 0.7, "WR": 0.5, "TE": 0.3, "RB": 0.2, "K": 0.1},
            game_share=0.4,
            defense=0.55,
            observations=[observation("same_team", "QB", "WR1", 0.35)],
            weeks=[1, 2, 3],
        )

    def test_converts_to_a_simulator_model(self):
        model = self.build().to_model()
        self.assertAlmostEqual(model.loading_for("QB"), 0.7)
        self.assertAlmostEqual(model.loading_for("WR"), 0.5)
        self.assertAlmostEqual(model.game_share, 0.4)

    def test_unknown_positions_fall_back(self):
        model = self.build().to_model()
        self.assertAlmostEqual(model.loading_for("DL"), model.offense)

    def test_serialises(self):
        payload = self.build().to_dict()
        self.assertIn("loadings", payload)
        self.assertIn("game_share", payload)
        self.assertEqual(len(payload["observations"]), 1)


class CorrelationModelTests(unittest.TestCase):
    def test_defaults_apply_without_fitted_loadings(self):
        model = CorrelationModel(offense=0.4, kicker=0.15)
        self.assertAlmostEqual(model.loading_for("WR"), 0.4)
        self.assertAlmostEqual(model.loading_for("K"), 0.15)

    def test_fitted_loadings_win_where_present(self):
        model = CorrelationModel(offense=0.4, loadings={"QB": 0.8})
        self.assertAlmostEqual(model.loading_for("QB"), 0.8)
        self.assertAlmostEqual(model.loading_for("RB"), 0.4)


def outlook(pid, position, team, opponent):
    return PlayerOutlook(
        player_id=pid,
        name=pid,
        position=position,
        nfl_team=team,
        opponent=opponent,
        mean=12.0,
        sd=6.0,
        play_prob=1.0,
    )


class StackIntensityTests(unittest.TestCase):
    def test_counts_same_team_pairs(self):
        stacked = [outlook("a", "QB", "BUF", "MIA"), outlook("b", "WR", "BUF", "MIA")]
        self.assertAlmostEqual(stack_intensity([(1, stacked)]), 1.0)

    def test_a_diversified_roster_scores_zero(self):
        spread = [outlook("a", "QB", "BUF", "MIA"), outlook("b", "WR", "KC", "DEN")]
        self.assertAlmostEqual(stack_intensity([(1, spread)]), 0.0)

    def test_three_on_one_team_is_two_pairs(self):
        triple = [outlook(str(i), "WR", "BUF", "MIA") for i in range(3)]
        self.assertAlmostEqual(stack_intensity([(1, triple)]), 2.0)

    def test_empty_input(self):
        self.assertEqual(stack_intensity([]), 0.0)


class TailComparisonTests(unittest.TestCase):
    def test_stronger_loadings_widen_a_stacked_roster(self):
        stacked = [outlook("s%d" % i, "WR", "BUF", "MIA") for i in range(6)]
        opponent = [outlook("o%d" % i, "WR", "T%d" % i, "X%d" % i) for i in range(6)]
        results = tail_comparison(
            [(1, stacked), (2, opponent)],
            [(1, 2)],
            [
                ("loose", CorrelationModel(offense=0.05, defense=0.05)),
                ("tight", CorrelationModel(offense=0.75, defense=0.05)),
            ],
            n_sims=4000,
            seed=2,
        )
        loose, tight = results
        self.assertGreater(tight.team_sd, loose.team_sd)
        self.assertGreater(tight.p95, loose.p95)
        self.assertLess(tight.p05, loose.p05)

    def test_probabilities_are_consistent(self):
        rosters = [outlook("a", "WR", "BUF", "MIA")], [outlook("b", "WR", "KC", "DEN")]
        results = tail_comparison(
            [(1, rosters[0]), (2, rosters[1])],
            [(1, 2)],
            [("default", CorrelationModel())],
            n_sims=2000,
            seed=1,
        )
        item = results[0]
        self.assertAlmostEqual(
            item.favorite_probability + item.upset_probability, 1.0, places=6
        )
        self.assertGreaterEqual(item.favorite_probability, 0.5)


if __name__ == "__main__":
    unittest.main()
