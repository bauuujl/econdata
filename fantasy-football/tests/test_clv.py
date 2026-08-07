import unittest

from ffbets.clv import ClvRecord, ClvReport, build_records, load_lines, synthetic_market
from ffbets.odds import american_to_decimal as a2d


class RecordTests(unittest.TestCase):
    def test_beating_the_close_is_positive_clv(self):
        record = ClvRecord(1, "A", "B", 0.6, bet_price=2.10, close_price=2.00)
        self.assertAlmostEqual(record.clv, 0.05)
        self.assertTrue(record.beat_close)

    def test_taking_a_worse_price_than_the_close(self):
        record = ClvRecord(1, "A", "B", 0.6, bet_price=1.90, close_price=2.00)
        self.assertAlmostEqual(record.clv, -0.05)
        self.assertFalse(record.beat_close)

    def test_no_close_means_no_grade(self):
        record = ClvRecord(1, "A", "B", 0.6, bet_price=2.0)
        self.assertIsNone(record.clv)
        self.assertIsNone(record.beat_close)
        self.assertIsNone(record.ev_at_close)

    def test_ev_at_close_is_zero_when_the_price_matches(self):
        record = ClvRecord(
            1, "A", "B", 0.6, bet_price=2.0, close_price=2.0, close_fair_probability=0.5
        )
        self.assertAlmostEqual(record.ev_at_close, 0.0)

    def test_ev_at_close_is_positive_when_you_beat_it(self):
        record = ClvRecord(
            1, "A", "B", 0.6, bet_price=2.2, close_price=2.0, close_fair_probability=0.5
        )
        self.assertGreater(record.ev_at_close, 0.0)

    def test_realised_profit(self):
        self.assertAlmostEqual(
            ClvRecord(1, "A", "B", 0.6, bet_price=2.5, won=True).realised, 1.5
        )
        self.assertAlmostEqual(
            ClvRecord(1, "A", "B", 0.6, bet_price=2.5, won=False).realised, -1.0
        )

    def test_probability_clv(self):
        record = ClvRecord(
            1, "A", "B", 0.6,
            bet_price=2.0,
            close_price=1.8,
            close_fair_probability=0.55,
            bet_fair_probability=0.50,
        )
        self.assertAlmostEqual(record.probability_clv, 0.05)


class ReportTests(unittest.TestCase):
    def build(self, prices):
        return ClvReport(
            records=[
                ClvRecord(
                    week=index + 1,
                    team="T%d" % index,
                    opponent="O%d" % index,
                    model_probability=0.55,
                    bet_price=bet_price,
                    close_price=close_price,
                    close_fair_probability=1.0 / close_price,
                    won=index % 2 == 0,
                )
                for index, (bet_price, close_price) in enumerate(prices)
            ]
        )

    def test_ungraded_records_are_excluded(self):
        report = ClvReport(
            records=[ClvRecord(1, "A", "B", 0.5, bet_price=2.0)]
        )
        self.assertEqual(report.count, 0)
        self.assertEqual(report.mean_clv, 0.0)

    def test_mean_clv_and_beat_rate(self):
        report = self.build([(2.1, 2.0), (2.1, 2.0), (1.9, 2.0), (2.0, 2.0)])
        self.assertAlmostEqual(report.mean_clv, (0.05 + 0.05 - 0.05 + 0.0) / 4)
        self.assertAlmostEqual(report.beat_close_rate, 0.5)

    def test_error_shrinks_with_more_bets(self):
        few = self.build([(2.1, 2.0), (1.9, 2.0)] * 3)
        many = self.build([(2.1, 2.0), (1.9, 2.0)] * 40)
        self.assertLess(many.clv_error, few.clv_error)

    def test_verdict_needs_a_sample(self):
        self.assertIn("too few", self.build([(2.1, 2.0)] * 5).verdict)

    def test_verdict_flags_a_clear_edge(self):
        # Mostly good prices with a little scatter, so the mean clears its own
        # standard error.
        prices = [(2.10, 2.0)] * 34 + [(2.02, 2.0)] * 6
        self.assertIn("clear of the noise", self.build(prices).verdict)

    def test_verdict_flags_a_negative_edge(self):
        prices = [(1.90, 2.0)] * 34 + [(1.98, 2.0)] * 6
        self.assertIn("negative CLV", self.build(prices).verdict)

    def test_verdict_handles_identical_prices(self):
        # Zero variance is not "no signal" -- it is a signal with no noise.
        self.assertIn("no variance", self.build([(2.1, 2.0)] * 40).verdict)
        self.assertIn("exactly zero", self.build([(2.0, 2.0)] * 40).verdict)


class LoadLinesTests(unittest.TestCase):
    def test_close_only_shorthand(self):
        lines = load_lines({"3": {"A": "-150"}})
        self.assertAlmostEqual(lines[3]["A"]["close"], a2d(-150))
        self.assertNotIn("open", lines[3]["A"])

    def test_open_and_close(self):
        lines = load_lines({"1": {"A": {"open": "+120", "close": "+105"}}})
        self.assertAlmostEqual(lines[1]["A"]["open"], a2d(120))
        self.assertAlmostEqual(lines[1]["A"]["close"], a2d(105))

    def test_bad_week_keys_are_skipped(self):
        self.assertEqual(load_lines({"notaweek": {"A": "-110"}}), {})

    def test_empty_payload(self):
        self.assertEqual(load_lines(None), {})


class BuildRecordsTests(unittest.TestCase):
    def test_devigs_both_sides_of_the_close(self):
        lines = load_lines(
            {
                "1": {
                    "A": {"open": "+110", "close": "-110"},
                    "B": {"open": "-130", "close": "-110"},
                }
            }
        )
        bets = [{"week": 1, "team": "A", "opponent": "B", "model_probability": 0.6, "won": True}]
        report = build_records(bets, lines, "shin")
        self.assertEqual(report.count, 1)
        record = report.records[0]
        self.assertAlmostEqual(record.close_fair_probability, 0.5, places=6)
        # Opened +110 and closed -110: a big move in the bettor's favour.
        self.assertGreater(record.clv, 0)

    def test_a_bet_with_no_line_is_dropped(self):
        bets = [{"week": 1, "team": "Z", "opponent": "Y", "model_probability": 0.6}]
        self.assertEqual(build_records(bets, load_lines({"1": {"A": "-110"}})).count, 0)


class SyntheticMarketTests(unittest.TestCase):
    def setUp(self):
        self.matchups = [
            {"week": 1 + index // 5, "home": "H%d" % index, "away": "A%d" % index,
             "fair_probability": 0.3 + 0.4 * (index % 5) / 4.0}
            for index in range(40)
        ]

    def test_every_price_is_a_valid_decimal_quote(self):
        lines = synthetic_market(self.matchups, seed=1)
        for teams in lines.values():
            for quotes in teams.values():
                for price in quotes.values():
                    self.assertGreater(price, 1.0)

    def test_extreme_probabilities_stay_quotable(self):
        extreme = [
            {"week": 1, "home": "H", "away": "A", "fair_probability": p}
            for p in (0.001, 0.999, 0.5)
        ]
        lines = synthetic_market(extreme, seed=2)
        for teams in lines.values():
            for quotes in teams.values():
                for price in quotes.values():
                    self.assertGreater(price, 1.0)

    def test_the_market_carries_a_margin(self):
        lines = synthetic_market(self.matchups, seed=3, close_noise=0.0, open_noise=0.0)
        entry = lines[1]
        pair = list(entry.values())[:2]
        overround = sum(1.0 / quote["close"] for quote in pair)
        self.assertGreater(overround, 1.0)

    def test_it_is_reproducible(self):
        self.assertEqual(
            synthetic_market(self.matchups, seed=5),
            synthetic_market(self.matchups, seed=5),
        )


if __name__ == "__main__":
    unittest.main()
