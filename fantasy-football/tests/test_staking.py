import unittest

from ffbets.staking import (
    Bet,
    StakingPolicy,
    blended_probability,
    kelly_fraction,
    miscalibrate,
    simulate_bankroll,
    stress_test,
)


def bet(week=1, probability=0.6, price=2.0, won=True, market=None):
    return Bet(
        week=week,
        label="t",
        model_probability=probability,
        price=price,
        won=won,
        market_probability=market,
    )


class KellyTests(unittest.TestCase):
    def test_textbook_case(self):
        self.assertAlmostEqual(kelly_fraction(0.6, 2.0), 0.20)

    def test_no_stake_without_an_edge(self):
        self.assertEqual(kelly_fraction(0.5, 2.0), 0.0)
        self.assertEqual(kelly_fraction(0.4, 2.0), 0.0)

    def test_invalid_price_is_no_bet(self):
        self.assertEqual(kelly_fraction(0.9, 1.0), 0.0)


class ShrinkageTests(unittest.TestCase):
    def test_shrinkage_moves_toward_the_market(self):
        self.assertAlmostEqual(blended_probability(0.70, 0.50, 0.30), 0.64)

    def test_zero_shrinkage_is_a_no_op(self):
        self.assertEqual(blended_probability(0.70, 0.50, 0.0), 0.70)

    def test_full_shrinkage_adopts_the_market(self):
        self.assertAlmostEqual(blended_probability(0.70, 0.50, 1.0), 0.50)

    def test_a_missing_market_leaves_the_model_alone(self):
        self.assertEqual(blended_probability(0.70, None, 0.5), 0.70)

    def test_shrinkage_shrinks_the_stake(self):
        item = bet(probability=0.70, market=0.50, price=2.0)
        plain = StakingPolicy("plain", kelly=1.0)
        shrunk = StakingPolicy("shrunk", kelly=1.0, shrinkage=0.5)
        self.assertLess(item.stake_fraction(shrunk), item.stake_fraction(plain))


class PolicyTests(unittest.TestCase):
    def test_the_cap_binds(self):
        item = bet(probability=0.95, price=3.0)
        capped = StakingPolicy("capped", kelly=1.0, cap=0.05)
        self.assertAlmostEqual(item.stake_fraction(capped), 0.05)

    def test_flat_staking_ignores_the_edge(self):
        policy = StakingPolicy("flat", flat=0.02)
        self.assertAlmostEqual(bet(probability=0.55).stake_fraction(policy), 0.02)
        self.assertAlmostEqual(bet(probability=0.95).stake_fraction(policy), 0.02)

    def test_flat_staking_still_needs_an_edge(self):
        policy = StakingPolicy("flat", flat=0.02)
        self.assertEqual(bet(probability=0.4, price=2.0).stake_fraction(policy), 0.0)

    def test_minimum_edge_filters_thin_bets(self):
        policy = StakingPolicy("picky", kelly=1.0, min_edge=0.05)
        self.assertEqual(bet(probability=0.51, price=2.0).stake_fraction(policy), 0.0)
        self.assertGreater(bet(probability=0.60, price=2.0).stake_fraction(policy), 0.0)

    def test_description_mentions_every_guard(self):
        policy = StakingPolicy("p", kelly=0.25, shrinkage=0.3, cap=0.05, weekly_cap=0.15)
        for fragment in ("Kelly", "market", "cap", "weekly"):
            self.assertIn(fragment, policy.description)


class BankrollTests(unittest.TestCase):
    def test_a_winning_bet_grows_the_bankroll(self):
        run = simulate_bankroll([bet(won=True, price=2.0)], StakingPolicy("p", flat=0.1))
        self.assertAlmostEqual(run.final, 1.1)
        self.assertEqual(run.bets_placed, 1)
        self.assertEqual(run.bets_won, 1)

    def test_a_losing_bet_shrinks_it(self):
        run = simulate_bankroll([bet(won=False, price=2.0)], StakingPolicy("p", flat=0.1))
        self.assertAlmostEqual(run.final, 0.9)

    def test_bets_in_one_week_are_sized_off_the_same_bankroll(self):
        bets = [bet(week=1, won=True), bet(week=1, won=True)]
        run = simulate_bankroll(bets, StakingPolicy("p", flat=0.1))
        # Both staked 0.1 of the starting 1.0, each returning +0.1.
        self.assertAlmostEqual(run.final, 1.2)

    def test_the_weekly_cap_scales_a_whole_week_down(self):
        bets = [bet(week=1, won=True) for _ in range(5)]
        policy = StakingPolicy("p", flat=0.1, weekly_cap=0.2)
        run = simulate_bankroll(bets, policy)
        self.assertAlmostEqual(run.staked, 0.2)

    def test_max_drawdown_measures_peak_to_trough(self):
        bets = [bet(week=1, won=True), bet(week=2, won=False), bet(week=3, won=False)]
        run = simulate_bankroll(bets, StakingPolicy("p", flat=0.5))
        self.assertGreater(run.max_drawdown, 0.0)
        self.assertLessEqual(run.max_drawdown, 1.0)

    def test_a_flat_path_has_no_drawdown(self):
        run = simulate_bankroll([bet(probability=0.3, price=2.0)], StakingPolicy("p"))
        self.assertEqual(run.bets_placed, 0)
        self.assertEqual(run.max_drawdown, 0.0)

    def test_roi_is_profit_over_turnover(self):
        run = simulate_bankroll([bet(won=True, price=2.0)], StakingPolicy("p", flat=0.1))
        self.assertAlmostEqual(run.roi, 1.0)

    def test_ruin_stops_the_walk(self):
        bets = [bet(week=w, won=False) for w in range(1, 20)]
        run = simulate_bankroll(bets, StakingPolicy("p", flat=0.9))
        self.assertLess(run.final, 0.05)
        self.assertLess(len(run.path), 20)


class MiscalibrationTests(unittest.TestCase):
    def test_a_factor_of_one_changes_nothing(self):
        self.assertAlmostEqual(miscalibrate(0.7, 1.0), 0.7, places=9)

    def test_stretching_pushes_away_from_a_half(self):
        self.assertGreater(miscalibrate(0.6, 1.6), 0.6)
        self.assertLess(miscalibrate(0.4, 1.6), 0.4)

    def test_a_coin_flip_stays_a_coin_flip(self):
        self.assertAlmostEqual(miscalibrate(0.5, 2.0), 0.5, places=9)


class StressTests(unittest.TestCase):
    def test_a_calibrated_bettor_never_bets(self):
        results = stress_test(overconfidence=1.0, seasons=30, weeks=6, seed=2)
        for item in results:
            self.assertAlmostEqual(item.median_final, 1.0, places=9)
            self.assertEqual(item.mean_bets, 0)

    def test_miscalibration_loses_money(self):
        results = stress_test(overconfidence=1.8, seasons=120, weeks=10, seed=2)
        # Every bet has negative true expected value, so no policy can profit.
        # Kelly-sized ones lose visibly; a tiny flat stake is near-neutral at
        # the median because its outcome distribution is barely moved at all.
        by_name = {item.policy.name: item for item in results}
        for name in ("Full Kelly", "Half Kelly", "Quarter Kelly"):
            self.assertLess(by_name[name].median_final, 1.0)
        for item in results:
            self.assertLess(item.median_final, 1.01)

    def test_smaller_fractions_lose_less_and_draw_down_less(self):
        results = {
            item.policy.name: item
            for item in stress_test(overconfidence=1.8, seasons=120, weeks=10, seed=2)
        }
        full = results["Full Kelly"]
        quarter = results["Quarter Kelly"]
        self.assertGreater(quarter.median_final, full.median_final)
        self.assertLess(quarter.mean_max_drawdown, full.mean_max_drawdown)

    def test_shrinkage_beats_plain_quarter_kelly_on_drawdown(self):
        results = {
            item.policy.name: item
            for item in stress_test(overconfidence=1.8, seasons=120, weeks=10, seed=2)
        }
        self.assertLess(
            results["Quarter + 30% market"].mean_max_drawdown,
            results["Quarter Kelly"].mean_max_drawdown,
        )

    def test_full_kelly_carries_real_ruin_risk(self):
        results = {
            item.policy.name: item
            for item in stress_test(overconfidence=1.8, seasons=150, weeks=12, seed=4)
        }
        self.assertGreater(results["Full Kelly"].ruin_rate, 0.0)
        self.assertEqual(results["Quarter + 30% market"].ruin_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
