"""Turn raw NFL statlines into fantasy points using a league's own rules.

Sleeper exposes a league's `scoring_settings` as a flat {stat_key: points}
map, and both the stats and projections feeds use those same stat keys.  So
scoring a statline is a dot product over the keys the league actually scores.

Doing it this way -- instead of reading the pre-computed `pts_ppr` field --
means custom scoring (TE premium, 6-point passing TDs, first-down bonuses,
bucketed defensive scoring) is handled for free.
"""

from __future__ import annotations

# Keys that show up in statline payloads but describe context rather than
# production.  A league would never score these, but a typo in a custom
# scoring setting shouldn't be able to turn snap counts into points.
NON_SCORING_KEYS = frozenset(
    {
        "gp",
        "gms_active",
        "off_snp",
        "def_snp",
        "st_snp",
        "tm_off_snp",
        "tm_def_snp",
        "tm_st_snp",
        "adp_dd_ppr",
        "pos_adp_dd_ppr",
        "pts_std",
        "pts_half_ppr",
        "pts_ppr",
        "pos_rank_std",
        "pos_rank_half_ppr",
        "pos_rank_ppr",
        "rank_std",
        "rank_half_ppr",
        "rank_ppr",
    }
)


def score_statline(stats, scoring_settings):
    """Fantasy points for one player-week.

    `stats` is a {stat_key: value} map from Sleeper's stats or projections
    feed; `scoring_settings` is the league's rule map.  Keys the league does
    not score contribute nothing.
    """
    if not stats:
        return 0.0
    total = 0.0
    for key, value in stats.items():
        if key in NON_SCORING_KEYS:
            continue
        weight = scoring_settings.get(key)
        if weight is None or value is None:
            continue
        total += float(value) * float(weight)
    return round(total, 2)


def score_week(week_stats, scoring_settings):
    """Score every player in one week's payload: {player_id: points}."""
    return {
        player_id: score_statline(stats, scoring_settings)
        for player_id, stats in week_stats.items()
    }


def scoring_label(scoring_settings):
    """Short human label for the league's receiving scoring ('PPR', etc.)."""
    rec = float(scoring_settings.get("rec", 0.0) or 0.0)
    if rec >= 1.0:
        base = "PPR"
    elif rec >= 0.4:
        base = "half-PPR"
    elif rec > 0:
        base = "%gpt-PPR" % rec
    else:
        base = "standard"

    extras = []
    if float(scoring_settings.get("pass_td", 4.0) or 0.0) >= 6.0:
        extras.append("6pt pass TD")
    te_bonus = float(scoring_settings.get("bonus_rec_te", 0.0) or 0.0)
    if te_bonus:
        extras.append("TE premium +%g" % te_bonus)
    if scoring_settings.get("rec_fd") or scoring_settings.get("rush_fd"):
        extras.append("first downs")
    return base + (" (%s)" % ", ".join(extras) if extras else "")
