import unittest

from ffbets import odds


class ConversionTests(unittest.TestCase):
    def test_american_to_decimal(self):
        self.assertAlmostEqual(odds.american_to_decimal(100), 2.0)
        self.assertAlmostEqual(odds.american_to_decimal(-110), 1.909090, places=5)
        self.assertAlmostEqual(odds.american_to_decimal(250), 3.5)
        self.assertAlmostEqual(odds.american_to_decimal(-200), 1.5)

    def test_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            odds.american_to_decimal(0)

    def test_round_trip(self):
        for price in (-500, -250, -110, 100, 145, 900):
            decimal = odds.american_to_decimal(price)
            self.assertEqual(odds.decimal_to_american(decimal), price)

    def test_decimal_to_american_crosses_at_even_money(self):
        self.assertEqual(odds.decimal_to_american(2.0), 100)
        self.assertEqual(odds.decimal_to_american(1.5), -200)

    def test_implied_probability(self):
        self.assertAlmostEqual(odds.implied_probability(-110), 0.523809, places=5)
        self.assertAlmostEqual(odds.implied_probability(100), 0.5)

    def test_probability_to_american_is_inverse(self):
        for probability in (0.25, 0.5, 0.62, 0.8):
            price = odds.probability_to_american(probability)
            self.assertAlmostEqual(odds.implied_probability(price), probability, places=3)

    def test_probability_bounds(self):
        for bad in (0.0, 1.0, -0.2, 1.4):
            with self.assertRaises(ValueError):
                odds.probability_to_american(bad)


class ParseTests(unittest.TestCase):
    def test_signed_strings_are_american(self):
        self.assertAlmostEqual(odds.parse_odds("+150"), 2.5)
        self.assertAlmostEqual(odds.parse_odds("-200"), 1.5)

    def test_unsigned_decimal_string_stays_decimal(self):
        self.assertAlmostEqual(odds.parse_odds("2.50"), 2.5)

    def test_unsigned_integer_string_is_american(self):
        self.assertAlmostEqual(odds.parse_odds("150"), 2.5)

    def test_numbers(self):
        self.assertAlmostEqual(odds.parse_odds(-200), 1.5)
        self.assertAlmostEqual(odds.parse_odds(1.91), 1.91)

    def test_rejects_junk(self):
        for bad in ("", "  ", "0.5"):
            with self.assertRaises(ValueError):
                odds.parse_odds(bad)


class VigTests(unittest.TestCase):
    def test_devig_sums_to_one(self):
        fair = odds.remove_vig([1.909090, 1.909090])
        self.assertAlmostEqual(sum(fair), 1.0)
        self.assertAlmostEqual(fair[0], 0.5)

    def test_devig_preserves_ordering(self):
        fair = odds.remove_vig([1.5, 3.0])
        self.assertAlmostEqual(sum(fair), 1.0)
        self.assertGreater(fair[0], fair[1])

    def test_hold_on_a_standard_two_way_market(self):
        # -110 both ways is the classic 4.5% hold.
        hold = odds.hold_percentage([odds.american_to_decimal(-110)] * 2)
        self.assertAlmostEqual(hold, 4.7619, places=3)

    def test_fair_market_has_no_hold(self):
        self.assertAlmostEqual(odds.hold_percentage([2.0, 2.0]), 0.0)


class StakingTests(unittest.TestCase):
    def test_expected_value_of_a_fair_bet_is_zero(self):
        self.assertAlmostEqual(odds.expected_value(0.5, 2.0), 0.0)

    def test_expected_value_signs(self):
        self.assertGreater(odds.expected_value(0.60, 2.0), 0)
        self.assertLess(odds.expected_value(0.40, 2.0), 0)

    def test_kelly_matches_the_textbook_case(self):
        # 60% at even money: stake edge/odds = 0.2/1.0 = 20% of bankroll.
        self.assertAlmostEqual(odds.kelly_fraction(0.6, 2.0, multiplier=1.0), 0.20)

    def test_kelly_multiplier_scales_the_stake(self):
        full = odds.kelly_fraction(0.6, 2.0, multiplier=1.0)
        quarter = odds.kelly_fraction(0.6, 2.0, multiplier=0.25)
        self.assertAlmostEqual(quarter, full * 0.25)

    def test_no_stake_without_an_edge(self):
        self.assertEqual(odds.kelly_fraction(0.45, 2.0), 0.0)
        self.assertEqual(odds.kelly_fraction(0.5, 2.0), 0.0)

    def test_stake_is_capped(self):
        self.assertLessEqual(odds.kelly_fraction(0.99, 5.0, multiplier=1.0, cap=0.25), 0.25)


if __name__ == "__main__":
    unittest.main()
