import unittest

from ffbets.scoring import score_statline, score_week, scoring_label

PPR = {
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "pass_int": -1.0,
    "rush_yd": 0.1,
    "rush_td": 6.0,
    "rec": 1.0,
    "rec_yd": 0.1,
    "rec_td": 6.0,
    "fum_lost": -2.0,
}


class ScoreStatlineTests(unittest.TestCase):
    def test_quarterback_line(self):
        line = {"pass_yd": 300, "pass_td": 2, "pass_int": 1, "rush_yd": 20}
        # 12 + 8 - 1 + 2
        self.assertAlmostEqual(score_statline(line, PPR), 21.0)

    def test_receiver_line_full_ppr(self):
        line = {"rec": 8, "rec_yd": 95, "rec_td": 1}
        self.assertAlmostEqual(score_statline(line, PPR), 23.5)

    def test_half_ppr_scores_receptions_at_half(self):
        half = dict(PPR, rec=0.5)
        line = {"rec": 8, "rec_yd": 95, "rec_td": 1}
        self.assertAlmostEqual(score_statline(line, half), 19.5)

    def test_unscored_stats_contribute_nothing(self):
        line = {"rec": 5, "rec_yd": 50, "tkl_ast": 40, "pass_cmp": 3}
        self.assertAlmostEqual(score_statline(line, PPR), 10.0)

    def test_context_keys_are_never_scored(self):
        # Even if a league somehow prices snap counts, they must not score.
        weird = dict(PPR, gp=100.0, off_snp=5.0, pts_ppr=99.0)
        self.assertAlmostEqual(
            score_statline({"gp": 1, "off_snp": 60, "pts_ppr": 30, "rec": 2}, weird), 2.0
        )

    def test_negative_and_empty_lines(self):
        self.assertAlmostEqual(score_statline({"fum_lost": 2}, PPR), -4.0)
        self.assertEqual(score_statline({}, PPR), 0.0)
        self.assertEqual(score_statline(None, PPR), 0.0)

    def test_none_values_are_skipped(self):
        self.assertAlmostEqual(score_statline({"rec": None, "rec_yd": 10}, PPR), 1.0)

    def test_bucketed_defensive_scoring(self):
        settings = {"sack": 1.0, "int": 2.0, "pts_allow_14_20": 1.0, "def_td": 6.0}
        line = {"sack": 3, "int": 1, "pts_allow": 17, "pts_allow_14_20": 1, "def_td": 0}
        # pts_allow itself is not a scored key; the bucket flag is.
        self.assertAlmostEqual(score_statline(line, settings), 6.0)


class ScoreWeekTests(unittest.TestCase):
    def test_scores_every_player(self):
        week = {"1": {"rec": 3, "rec_yd": 30}, "2": {"rush_td": 1}}
        self.assertEqual(score_week(week, PPR), {"1": 6.0, "2": 6.0})


class LabelTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(scoring_label(PPR), "PPR")
        self.assertEqual(scoring_label(dict(PPR, rec=0.5)), "half-PPR")
        self.assertEqual(scoring_label(dict(PPR, rec=0.0)), "standard")

    def test_extras_are_called_out(self):
        label = scoring_label(dict(PPR, pass_td=6.0, bonus_rec_te=0.5))
        self.assertIn("6pt pass TD", label)
        self.assertIn("TE premium", label)


if __name__ == "__main__":
    unittest.main()
