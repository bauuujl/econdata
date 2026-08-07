"""Rendering: a terminal report and a self-contained HTML report."""

from __future__ import annotations

import html
import math

RULE = "-" * 78


def signed(value, digits=1):
    return "%+.*f" % (digits, value)


def american(value):
    return "%+d" % value


def percent(value, digits=1):
    return "%.*f%%" % (digits, 100.0 * value)


# ----------------------------------------------------------------------
# terminal
# ----------------------------------------------------------------------
def render_text(analysis, detail=False, top=8):
    lines = []
    out = lines.append

    out("=" * 78)
    out("  %s  |  %s Week %s" % (analysis.league_name, analysis.season, analysis.week))
    out(
        "  %s scoring  |  %s simulations  |  %s lineups"
        % (analysis.scoring, "{:,}".format(analysis.n_sims), analysis.lineup_mode)
    )
    out("=" * 78)

    for note in analysis.notes:
        out("  note: %s" % note)
    if analysis.notes:
        out("")

    out("")
    out("MATCHUPS")
    out(RULE)
    ranked = sorted(analysis.matchups, key=lambda m: m.confidence, reverse=True)
    for matchup in ranked:
        for team, probability in (
            (matchup.home, matchup.p_home),
            (matchup.away, matchup.p_away),
        ):
            marker = ">" if probability == matchup.favorite_probability else " "
            out(
                "%s %-26.26s %6.1f pts  +/- %4.1f   win %6s   fair %6s"
                % (
                    marker,
                    team.name,
                    team.mean,
                    team.sd,
                    percent(probability),
                    american(
                        _fair_price(probability, matchup.p_home + matchup.p_away)
                    ),
                )
            )
        median, low, high = matchup.favorite_margin
        out(
            "  margin: %s by %.1f (median); 80%% of outcomes %s to %s for them%s"
            % (
                matchup.favorite.name,
                median,
                signed(low),
                signed(high),
                "   tie %s" % percent(matchup.p_tie) if matchup.p_tie > 0.001 else "",
            )
        )
        if detail:
            out("")
            out(_lineup_block(matchup.home))
            out(_lineup_block(matchup.away))
        out("")

    priced = [item for item in analysis.recommendations if item.priced]
    if priced:
        out("")
        out("BEST BETS  (vs your sportsbook, %g x Kelly)" % analysis.kelly_multiplier)
        out(RULE)
        out(
            "%-24s %7s %7s %8s %9s %8s %7s"
            % ("SIDE", "BOOK", "FAIR", "MODEL", "MARKET", "EV/$1", "STAKE")
        )
        for item in priced[:top]:
            baseline = (
                item.devig_probability
                if item.devig_probability is not None
                else item.implied_probability
            )
            out(
                "%-24.24s %7s %7s %8s %9s %8s %7s"
                % (
                    item.team,
                    american(item.price_american),
                    american(item.fair_american),
                    percent(item.model_probability),
                    percent(baseline),
                    "%+.3f" % item.expected_value,
                    percent(item.kelly) if item.kelly else "-",
                )
            )
        out("")
        playable = [item for item in priced if item.expected_value > 0]
        if playable:
            best = playable[0]
            out(
                "  -> Best value: %s at %s (model %s vs market %s, EV %+.1f%% per $1)"
                % (
                    best.team,
                    american(best.price_american),
                    percent(best.model_probability),
                    percent(
                        best.devig_probability
                        if best.devig_probability is not None
                        else best.implied_probability
                    ),
                    100 * best.expected_value,
                )
            )
            hold = [item.market_hold for item in playable if item.market_hold]
            if hold:
                out("     Book hold on that market: %.1f%%" % hold[0])
        else:
            out("  -> No priced side clears the vig. The disciplined play is no bet.")
    else:
        out("")
        out("MODEL LEANS  (no book odds supplied -- pass --odds to price them)")
        out(RULE)
        out("%-26s %-24s %9s %8s" % ("BET ON", "AGAINST", "WIN PROB", "FAIR"))
        for item in analysis.recommendations[:top]:
            if item.model_probability < 0.5:
                continue
            out(
                "%-26.26s %-24.24s %9s %8s"
                % (
                    item.team,
                    item.opponent,
                    percent(item.model_probability),
                    american(item.fair_american),
                )
            )
        out("")
        out(
            "  Win probabilities carry a Monte Carlo error of about %s."
            % percent(max((i.error for i in analysis.recommendations), default=0.0), 2)
        )

    out("")
    out(RULE)
    out(
        "  Projections are estimates, not outcomes. A 65% favourite still loses "
        "about one week in three."
    )
    return "\n".join(lines)


def _fair_price(probability, decisive):
    from .odds import probability_to_american

    value = probability / decisive if decisive > 0 else 0.5
    return probability_to_american(min(max(value, 1e-4), 1 - 1e-4))


def _lineup_block(team):
    rows = ["    %s starters:" % team.name]
    for player in sorted(team.starters, key=lambda p: p.expected, reverse=True):
        rows.append(
            "      %-4s %-22.22s %-4s %5.1f  +/-%4.1f  %s"
            % (
                player.position,
                player.name,
                player.nfl_team or "--",
                player.expected,
                player.sd,
                player.flags or player.source,
            )
        )
    return "\n".join(rows)


# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------
CSS = """
:root{--bg:#ffffff;--fg:#16181d;--muted:#6b7280;--line:#e4e7ec;--card:#f8fafc;
--accent:#2563eb;--good:#047857;--bad:#b91c1c;--bar:#dbe4f5}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e8eaed;--muted:#9aa2ad;
--line:#272b33;--card:#171a20;--accent:#7aa2f7;--good:#4ade80;--bad:#f87171;--bar:#232a38}}
:root[data-theme=dark]{--bg:#0f1115;--fg:#e8eaed;--muted:#9aa2ad;--line:#272b33;
--card:#171a20;--accent:#7aa2f7;--good:#4ade80;--bad:#f87171;--bar:#232a38}
:root[data-theme=light]{--bg:#ffffff;--fg:#16181d;--muted:#6b7280;--line:#e4e7ec;
--card:#f8fafc;--accent:#2563eb;--good:#047857;--bad:#b91c1c;--bar:#dbe4f5}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:40px 20px 72px}
h1{font-size:1.55rem;margin:0 0 4px;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:.9rem;margin-bottom:28px}
h2{font-size:.78rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
margin:38px 0 12px;font-weight:600}
.note{background:var(--card);border-left:3px solid var(--accent);padding:8px 12px;
margin:6px 0;font-size:.85rem;color:var(--muted);border-radius:0 4px 4px 0}
.game{border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:12px;
background:var(--card)}
.side{display:flex;align-items:center;gap:12px;padding:5px 0}
.nm{flex:1;min-width:0;font-weight:600;overflow:hidden;text-overflow:ellipsis;
white-space:nowrap}
.nm.dog{font-weight:400;color:var(--muted)}
.pts{font-variant-numeric:tabular-nums;font-size:.86rem;color:var(--muted);width:104px;
text-align:right;flex:none}
.wp{font-variant-numeric:tabular-nums;font-weight:600;width:62px;text-align:right;flex:none}
.odd{font-variant-numeric:tabular-nums;width:60px;text-align:right;flex:none;
color:var(--muted);font-size:.86rem}
.track{height:6px;background:var(--bar);border-radius:3px;overflow:hidden;margin:8px 0 4px}
.fill{height:100%;background:var(--accent);border-radius:3px}
.margin{font-size:.82rem;color:var(--muted);font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.88rem;min-width:640px}
th{text-align:right;font-size:.7rem;text-transform:uppercase;letter-spacing:.07em;
color:var(--muted);border-bottom:1px solid var(--line);padding:8px 10px;font-weight:600}
th:first-child,td:first-child{text-align:left}
td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;
font-variant-numeric:tabular-nums}
tr.play td:first-child{font-weight:600}
.pos{color:var(--good)}.neg{color:var(--bad)}
.verdict{border:1px solid var(--line);border-left:3px solid var(--good);background:var(--card);
border-radius:0 8px 8px 0;padding:14px 16px;margin-top:16px}
.verdict.none{border-left-color:var(--muted)}
.verdict.alarm{border-left-color:var(--bad);border-left-width:4px}
.warn{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--bad);
border-radius:0 8px 8px 0;padding:13px 16px;margin:14px 0;font-size:.9rem}
footer{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);
color:var(--muted);font-size:.8rem}
"""


def render_html(analysis, title=None):
    e = html.escape
    parts = []
    add = parts.append

    heading = title or "%s - Week %s" % (analysis.league_name, analysis.week)
    add("<title>%s</title>" % e(heading))
    add("<style>%s</style>" % CSS)
    add('<div class="wrap">')
    add("<h1>%s</h1>" % e(heading))
    add(
        '<div class="sub">%s season, week %s &middot; %s scoring &middot; %s simulations &middot; %s lineups</div>'
        % (
            e(analysis.season),
            analysis.week,
            e(analysis.scoring),
            "{:,}".format(analysis.n_sims),
            e(analysis.lineup_mode),
        )
    )

    for note in analysis.notes:
        add('<div class="note">%s</div>' % e(note))

    priced = [item for item in analysis.recommendations if item.priced]
    if priced:
        add("<h2>Best bets</h2>")
        add('<div class="scroll"><table>')
        add(
            "<tr><th>Side</th><th>Book</th><th>Fair</th><th>Model</th>"
            "<th>Market</th><th>EV / $1</th><th>Stake</th></tr>"
        )
        for item in priced:
            baseline = (
                item.devig_probability
                if item.devig_probability is not None
                else item.implied_probability
            )
            klass = "pos" if item.expected_value > 0 else "neg"
            add(
                '<tr class="%s"><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
                '<td>%s</td><td class="%s">%+.3f</td><td>%s</td></tr>'
                % (
                    "play" if item.expected_value > 0 else "",
                    e(item.team),
                    american(item.price_american),
                    american(item.fair_american),
                    percent(item.model_probability),
                    percent(baseline),
                    klass,
                    item.expected_value,
                    percent(item.kelly) if item.kelly else "&mdash;",
                )
            )
        add("</table></div>")

        playable = [item for item in priced if item.expected_value > 0]
        if playable:
            best = playable[0]
            add(
                '<div class="verdict"><strong>Best value: %s at %s.</strong> '
                "The model gives them %s to win; the de-vigged market implies %s. "
                "That is %+.1f%% expected return per dollar, or %s of bankroll at "
                "%g&times; Kelly.</div>"
                % (
                    e(best.team),
                    american(best.price_american),
                    percent(best.model_probability),
                    percent(
                        best.devig_probability
                        if best.devig_probability is not None
                        else best.implied_probability
                    ),
                    100 * best.expected_value,
                    percent(best.kelly),
                    analysis.kelly_multiplier,
                )
            )
        else:
            add(
                '<div class="verdict none"><strong>No bet.</strong> Every priced '
                "side sits behind the book's margin once the vig is removed.</div>"
            )
    else:
        add("<h2>Model leans</h2>")
        add('<div class="scroll"><table>')
        add("<tr><th>Bet on</th><th>Against</th><th>Win probability</th><th>Fair price</th></tr>")
        for item in analysis.recommendations:
            if item.model_probability < 0.5:
                continue
            add(
                "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                % (
                    e(item.team),
                    e(item.opponent),
                    percent(item.model_probability),
                    american(item.fair_american),
                )
            )
        add("</table></div>")

    add("<h2>Every matchup</h2>")
    for matchup in sorted(analysis.matchups, key=lambda m: m.confidence, reverse=True):
        add('<div class="game">')
        decisive = matchup.p_home + matchup.p_away
        for team, probability in (
            (matchup.home, matchup.p_home),
            (matchup.away, matchup.p_away),
        ):
            favourite = probability == matchup.favorite_probability
            add('<div class="side">')
            add('<div class="nm%s">%s</div>' % ("" if favourite else " dog", e(team.name)))
            add('<div class="pts">%.1f &plusmn; %.1f</div>' % (team.mean, team.sd))
            add('<div class="wp">%s</div>' % percent(probability))
            add('<div class="odd">%s</div>' % american(_fair_price(probability, decisive)))
            add("</div>")
        add(
            '<div class="track"><div class="fill" style="width:%.1f%%"></div></div>'
            % (100.0 * matchup.p_home)
        )
        median, low, high = matchup.favorite_margin
        add(
            '<div class="margin">Median margin: %s by %.1f &middot; 80%% of outcomes '
            "land between %s and %s in their favour</div>"
            % (e(matchup.favorite.name), median, signed(low), signed(high))
        )
        add("</div>")

    add(
        "<footer>Projections are estimates, not outcomes &mdash; a 65% favourite still "
        "loses about one week in three, and the edge shown is only as good as the "
        "inputs behind it. Bet only what you can afford to lose.</footer>"
    )
    add("</div>")
    return "\n".join(parts)
