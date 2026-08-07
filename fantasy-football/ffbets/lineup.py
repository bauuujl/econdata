"""Roster slot eligibility and optimal lineup construction."""

from __future__ import annotations

# Which positions may fill each roster slot.  Anything not listed here is
# treated as an exact-position slot (QB fills QB, and so on).
FLEX_ELIGIBILITY = {
    "FLEX": {"RB", "WR", "TE"},
    "WRRB_FLEX": {"RB", "WR"},
    "WRRB_WRT": {"RB", "WR", "TE"},
    "REC_FLEX": {"WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
    "IDP_FLEX": {"DL", "LB", "DB"},
    "DEF_FLEX": {"DL", "LB", "DB"},
}

# Slots that never score.
NON_SCORING_SLOTS = {"BN", "IR", "TAXI"}


def slot_eligibility(slot):
    """The set of positions that can start in `slot`."""
    return FLEX_ELIGIBILITY.get(slot, {slot})


def starting_slots(roster_positions):
    """Just the slots that put points on the board, in league order."""
    return [slot for slot in roster_positions if slot not in NON_SCORING_SLOTS]


def optimal_lineup(outlooks, roster_positions):
    """Pick the highest-expected-points legal lineup from a roster.

    Slots are filled most-restrictive first, so a QB is never burned in a
    SUPER_FLEX while the QB slot sits empty.  That ordering is provably
    optimal when slot eligibility is nested (QB inside SUPER_FLEX, RB inside
    FLEX inside SUPER_FLEX), which covers every standard league; exotic
    combinations of partially-overlapping flexes can in principle do slightly
    better with a full assignment solve.

    Returns (starters, bench) where both are lists of PlayerOutlook.
    """
    slots = starting_slots(roster_positions)
    available = sorted(outlooks, key=lambda player: player.expected, reverse=True)

    order = sorted(
        range(len(slots)), key=lambda index: (len(slot_eligibility(slots[index])), index)
    )

    chosen = {}
    used = set()
    for index in order:
        eligible = slot_eligibility(slots[index])
        for player in available:
            if player.player_id in used:
                continue
            if player.position in eligible:
                chosen[index] = player
                used.add(player.player_id)
                break

    starters = [chosen[index] for index in range(len(slots)) if index in chosen]
    bench = [player for player in outlooks if player.player_id not in used]
    return starters, bench


def declared_lineup(outlooks, starter_ids):
    """The lineup the manager actually set, in slot order."""
    by_id = {player.player_id: player for player in outlooks}
    return [by_id[pid] for pid in starter_ids if pid in by_id]
