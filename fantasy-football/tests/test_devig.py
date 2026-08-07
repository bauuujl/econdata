import unittest

from ffbets import devig
from ffbets.odds import american_to_decimal as a2d


class BasicsTests(unittest.TestCase):
    def test_every_method_sums_to_one(self):
        for prices in ([a2d(-110)] * 2, [a2d(-300), a2d(240)], [a2d(150), a2d(220), a2d(260)]):
            for method in devig.METHODS:
                self.assertAlmostEqual(sum(devig.remove_vig(prices, method)), 1.0, places=9)

    def test_a_fair_market_is_left_alone(self):
        for method in devig.METHODS:
            fair = devig.remove_vig([2.0, 2.0], method)
            self.assertAlmostEqual(fair[0], 0.5, places=6)

    def test_symmetric_market_is_symmetric_under_every_method(self):
        for method in devig.METHODS:
            fair = devig.remove_vig([a2d(-110)] * 2, method)
            self.assertAlmostEqual(fair[0], fair[1], places=9)

    def test_ordering_is_preserved(self):
        for method in devig.METHODS:
            fair = devig.remove_vig([a2d(-300), a2d(240)], method)
            self.assertGreater(fair[0], fair[1])

    def test_overround_and_hold(self):
        prices = [a2d(-110)] * 2
        self.assertAlmostEqual(devig.overround(prices), 1.047619, places=5)
        self.assertAlmostEqual(devig.hold_percentage(prices), 4.7619, places=3)

    def test_multiplicative_is_an_alias_for_proportional(self):
        prices = [a2d(-300), a2d(240)]
        self.assertEqual(
            devig.remove_vig(prices, "multiplicative"),
            devig.remove_vig(prices, "proportional"),
        )

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            devig.remove_vig([1.9], "shin")
        with self.assertRaises(ValueError):
            devig.remove_vig([0.9, 2.0], "shin")
        with self.assertRaises(ValueError):
            devig.remove_vig([2.0, 2.0], "nonsense")


class FavouriteBiasTests(unittest.TestCase):
    """The whole reason for offering a choice of method."""

    def setUp(self):
        self.prices = [a2d(-300), a2d(240)]
        self.fair = devig.compare_methods(self.prices)

    def test_proportional_gives_the_favourite_the_least(self):
        # Proportional strips margin in proportion to probability, so it takes
        # the most (in absolute terms) off the favourite. That makes it the
        # method that UNDERSTATES favourites relative to the others.
        favourite = {name: values[0] for name, values in self.fair.items()}
        self.assertEqual(min(favourite, key=favourite.get), "proportional")

    def test_power_gives_the_favourite_the_most(self):
        favourite = {name: values[0] for name, values in self.fair.items()}
        self.assertEqual(max(favourite, key=favourite.get), "power")

    def test_shin_sits_between_proportional_and_power(self):
        self.assertGreater(self.fair["shin"][0], self.fair["proportional"][0])
        self.assertLess(self.fair["shin"][0], self.fair["power"][0])

    def test_the_methods_disagree_enough_to_matter(self):
        favourite = [values[0] for values in self.fair.values()]
        self.assertGreater(max(favourite) - min(favourite), 0.01)

    def test_disagreement_grows_with_how_lopsided_the_market_is(self):
        def spread(prices):
            values = [v[0] for v in devig.compare_methods(prices).values()]
            return max(values) - min(values)

        self.assertGreater(spread([a2d(-1200), a2d(750)]), spread([a2d(-130), a2d(110)]))


class ShinTests(unittest.TestCase):
    def test_insider_fraction_is_zero_on_a_fair_book(self):
        self.assertEqual(devig.shin_insider_fraction([2.0, 2.0]), 0.0)

    def test_insider_fraction_rises_with_the_margin(self):
        thin = devig.shin_insider_fraction([a2d(-105), a2d(-105)])
        fat = devig.shin_insider_fraction([a2d(-130), a2d(-130)])
        self.assertGreater(fat, thin)
        self.assertLess(fat, 1.0)

    def test_a_book_with_no_overround_falls_back_cleanly(self):
        fair = devig.remove_vig([2.2, 2.2], "shin")
        self.assertAlmostEqual(sum(fair), 1.0, places=9)


class AdditiveTests(unittest.TestCase):
    def test_subtracts_the_margin_evenly(self):
        fair = devig.remove_vig([a2d(-300), a2d(240)], "additive")
        raw = [1 / a2d(-300), 1 / a2d(240)]
        self.assertAlmostEqual(raw[0] - fair[0], raw[1] - fair[1], places=9)

    def test_never_returns_a_negative_probability(self):
        # An extreme market would drive the longshot below zero under a naive
        # even split.
        fair = devig.remove_vig([1.01, 25.0], "additive")
        self.assertTrue(all(value > 0 for value in fair))
        self.assertAlmostEqual(sum(fair), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
