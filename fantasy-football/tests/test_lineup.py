import unittest

from ffbets.lineup import (
    declared_lineup,
    optimal_lineup,
    slot_eligibility,
    starting_slots,
)
from ffbets.projections import PlayerOutlook


def player(player_id, position, mean, play_prob=1.0):
    return PlayerOutlook(
        player_id=player_id,
        name=player_id,
        position=position,
        mean=mean,
        sd=mean * 0.5,
        play_prob=play_prob,
    )


class SlotTests(unittest.TestCase):
    def test_exact_slots(self):
        self.assertEqual(slot_eligibility("QB"), {"QB"})

    def test_flex_slots(self):
        self.assertEqual(slot_eligibility("FLEX"), {"RB", "WR", "TE"})
        self.assertEqual(slot_eligibility("SUPER_FLEX"), {"QB", "RB", "WR", "TE"})

    def test_bench_is_excluded(self):
        positions = ["QB", "RB", "FLEX", "BN", "BN", "IR"]
        self.assertEqual(starting_slots(positions), ["QB", "RB", "FLEX"])


class OptimalLineupTests(unittest.TestCase):
    def test_picks_the_best_legal_lineup(self):
        roster = [
            player("qb1", "QB", 22),
            player("rb1", "RB", 18),
            player("rb2", "RB", 9),
            player("wr1", "WR", 15),
            player("te1", "TE", 4),
        ]
        starters, bench = optimal_lineup(roster, ["QB", "RB", "WR", "FLEX", "BN"])
        self.assertEqual(
            sorted(p.player_id for p in starters), ["qb1", "rb1", "rb2", "wr1"]
        )
        self.assertEqual([p.player_id for p in bench], ["te1"])

    def test_flex_takes_the_best_leftover(self):
        roster = [
            player("rb1", "RB", 20),
            player("rb2", "RB", 14),
            player("te1", "TE", 16),
        ]
        starters, _ = optimal_lineup(roster, ["RB", "FLEX"])
        self.assertEqual(sorted(p.player_id for p in starters), ["rb1", "te1"])

    def test_quarterback_is_not_wasted_in_superflex(self):
        # The restrictive QB slot must be filled first, or the lone QB ends up
        # in SUPER_FLEX and the QB slot goes empty.
        roster = [
            player("qb1", "QB", 25),
            player("rb1", "RB", 12),
            player("wr1", "WR", 11),
        ]
        starters, _ = optimal_lineup(roster, ["QB", "RB", "SUPER_FLEX"])
        self.assertEqual(len(starters), 3)
        self.assertEqual(sorted(p.player_id for p in starters), ["qb1", "rb1", "wr1"])

    def test_injured_players_slide_down_the_board(self):
        roster = [
            player("healthy", "RB", 12),
            player("hurt", "RB", 20, play_prob=0.0),
        ]
        starters, _ = optimal_lineup(roster, ["RB"])
        self.assertEqual(starters[0].player_id, "healthy")

    def test_short_roster_leaves_slots_empty(self):
        starters, bench = optimal_lineup([player("qb1", "QB", 20)], ["QB", "RB", "WR"])
        self.assertEqual(len(starters), 1)
        self.assertEqual(bench, [])

    def test_no_player_is_started_twice(self):
        roster = [player("rb1", "RB", 20), player("wr1", "WR", 10)]
        starters, _ = optimal_lineup(roster, ["RB", "FLEX", "FLEX"])
        ids = [p.player_id for p in starters]
        self.assertEqual(len(ids), len(set(ids)))


class DeclaredLineupTests(unittest.TestCase):
    def test_keeps_submitted_order_and_drops_unknowns(self):
        roster = [player("a", "QB", 20), player("b", "RB", 10)]
        starters = declared_lineup(roster, ["b", "a", "ghost"])
        self.assertEqual([p.player_id for p in starters], ["b", "a"])


if __name__ == "__main__":
    unittest.main()
