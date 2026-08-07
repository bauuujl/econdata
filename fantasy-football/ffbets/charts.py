"""Dependency-free inline SVG charts for the HTML report.

No plotting library is available and none is wanted -- the report has to stay
a single self-contained file. These build SVG by hand.

Both charts are drawn against CSS custom properties rather than literal hex,
so light and dark mode swap in one place. The categorical hues are the first
slots of a palette validated for colour-vision deficiency; the series count
is capped where that validation stops holding. Every series is also directly
labelled and repeated in a table underneath, so colour never carries meaning
on its own.
"""

from __future__ import annotations

import html

# Categorical slots, in validated order. Light and dark are the same hues
# stepped for their own surface, not an automatic flip.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]

CHART_CSS = """
.viz{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz{
--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300}}
:root[data-theme="dark"] .viz{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300}
.viz{background:var(--surface);border:1px solid var(--grid);border-radius:10px;
padding:14px 16px 10px;margin:14px 0}
.viz svg{display:block;width:100%;height:auto;overflow:visible}
.viz .gl{stroke:var(--grid);stroke-width:1}
.viz .ax{stroke:var(--axis);stroke-width:1}
.viz .tick{fill:var(--muted);font-size:10px;font-variant-numeric:tabular-nums}
.viz .alab{fill:var(--ink2);font-size:11px}
.viz .ref{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 4;opacity:.7}
.viz .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.viz .mk{stroke:var(--surface);stroke-width:2}
.viz .eb{stroke-width:1.5;opacity:.45}
.viz .dl{font-size:11px;font-weight:600}
.viz .cap{fill:var(--ink2);font-size:11px}
.viz-title{font-size:.82rem;font-weight:600;color:var(--ink);margin:0 0 2px}
.viz-sub{font-size:.76rem;color:var(--ink2);margin:0 0 10px}
.viz-legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:8px;font-size:.76rem;
color:var(--ink2)}
.viz-legend span{display:inline-flex;align-items:center;gap:6px}
.viz-legend i{width:11px;height:3px;border-radius:2px;display:inline-block}
"""


def _escape(text):
    return html.escape(str(text), quote=True)


def _chrome(width, height, pad):
    """Plot-area geometry shared by both charts."""
    left, right, top, bottom = pad
    return {
        "x0": left,
        "y0": top,
        "x1": width - right,
        "y1": height - bottom,
        "w": width - left - right,
        "h": height - top - bottom,
    }


def reliability_chart(curves, title="Reliability", subtitle="", width=560, height=380):
    """Predicted vs realised win rate, with a perfect-calibration diagonal.

    `curves` is [(label, [ReliabilityBin, ...]), ...]. Points on the diagonal
    are perfectly calibrated; points below it mean the forecast was too high,
    which is what overconfidence looks like.
    """
    box = _chrome(width, height, (46, 92, 14, 40))
    parts = []
    add = parts.append

    def px(value):
        return box["x0"] + value * box["w"]

    def py(value):
        return box["y1"] - value * box["h"]

    add('<div class="viz">')
    add('<p class="viz-title">%s</p>' % _escape(title))
    if subtitle:
        add('<p class="viz-sub">%s</p>' % _escape(subtitle))
    add(
        '<svg viewBox="0 0 %d %d" role="img" aria-label="%s">'
        % (width, height, _escape(title + ". " + subtitle))
    )

    # Grid and axes.
    for step in range(5):
        value = step / 4.0
        add(
            '<line class="gl" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>'
            % (box["x0"], py(value), box["x1"], py(value))
        )
        add(
            '<text class="tick" x="%.1f" y="%.1f" text-anchor="end">%d%%</text>'
            % (box["x0"] - 6, py(value) + 3, round(100 * value))
        )
        add(
            '<text class="tick" x="%.1f" y="%.1f" text-anchor="middle">%d%%</text>'
            % (px(value), box["y1"] + 15, round(100 * value))
        )
    add(
        '<line class="ax" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>'
        % (box["x0"], box["y1"], box["x1"], box["y1"])
    )

    # Perfect calibration.
    add(
        '<line class="ref" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"><title>'
        "Perfect calibration</title></line>"
        % (px(0), py(0), px(1), py(1))
    )
    add(
        '<text class="cap" x="%.1f" y="%.1f" text-anchor="end">perfect</text>'
        % (px(0.97), py(0.97) - 6)
    )

    add(
        '<text class="alab" x="%.1f" y="%.1f" text-anchor="middle">forecast probability</text>'
        % ((box["x0"] + box["x1"]) / 2, height - 6)
    )
    add(
        '<text class="alab" transform="translate(11,%.1f) rotate(-90)" '
        'text-anchor="middle">observed win rate</text>' % ((box["y0"] + box["y1"]) / 2)
    )

    for index, (label, curve) in enumerate(curves[:6]):
        colour = "var(--s%d)" % (index + 1)
        if not curve:
            continue

        # Wilson intervals: the sample is small and the reader needs to see it.
        for item in curve:
            add(
                '<line class="eb" stroke="%s" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>'
                % (
                    colour,
                    px(item.predicted),
                    py(item.interval_low),
                    px(item.predicted),
                    py(item.interval_high),
                )
            )

        points = " ".join(
            "%.1f,%.1f" % (px(item.predicted), py(item.actual)) for item in curve
        )
        add('<polyline class="ln" stroke="%s" points="%s"/>' % (colour, points))

        for item in curve:
            add(
                '<circle class="mk" fill="%s" cx="%.1f" cy="%.1f" r="4.5">'
                "<title>%s: forecast %.0f%%, observed %.0f%% (%d matchups, "
                "95%% interval %.0f-%.0f%%)</title></circle>"
                % (
                    colour,
                    px(item.predicted),
                    py(item.actual),
                    _escape(label),
                    100 * item.predicted,
                    100 * item.actual,
                    item.count,
                    100 * item.interval_low,
                    100 * item.interval_high,
                )
            )

        last = curve[-1]
        add(
            '<text class="dl" fill="%s" x="%.1f" y="%.1f">%s</text>'
            % (colour, px(last.predicted) + 8, py(last.actual) + 4, _escape(label))
        )

    add("</svg>")
    add('<div class="viz-legend">')
    for index, (label, _) in enumerate(curves[:6]):
        add(
            '<span><i style="background:var(--s%d)"></i>%s</span>'
            % (index + 1, _escape(label))
        )
    add("</div></div>")
    return "\n".join(parts)


def bankroll_chart(runs, title="Bankroll", subtitle="", width=620, height=360):
    """Bankroll paths over the season, one line per staking policy.

    `runs` is [(label, [values...]), ...] starting at 1.0.
    """
    series = [(label, path) for label, path in runs if path][:6]
    if not series:
        return ""

    length = max(len(path) for _, path in series)
    highest = max(max(path) for _, path in series)
    lowest = min(min(path) for _, path in series)
    top = max(highest * 1.05, 1.05)
    bottom = min(lowest * 0.95, 0.95)
    span = top - bottom or 1.0

    box = _chrome(width, height, (46, 132, 14, 38))
    parts = []
    add = parts.append

    def px(index):
        return box["x0"] + (index / max(length - 1, 1)) * box["w"]

    def py(value):
        return box["y1"] - ((value - bottom) / span) * box["h"]

    add('<div class="viz">')
    add('<p class="viz-title">%s</p>' % _escape(title))
    if subtitle:
        add('<p class="viz-sub">%s</p>' % _escape(subtitle))
    add(
        '<svg viewBox="0 0 %d %d" role="img" aria-label="%s">'
        % (width, height, _escape(title + ". " + subtitle))
    )

    for step in range(5):
        value = bottom + span * step / 4.0
        add(
            '<line class="gl" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>'
            % (box["x0"], py(value), box["x1"], py(value))
        )
        add(
            '<text class="tick" x="%.1f" y="%.1f" text-anchor="end">%.2f</text>'
            % (box["x0"] - 6, py(value) + 3, value)
        )

    # The break-even line is the one the reader is actually judging against.
    if bottom < 1.0 < top:
        add(
            '<line class="ref" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"><title>'
            "Break even</title></line>" % (box["x0"], py(1.0), box["x1"], py(1.0))
        )

    for step in range(0, length, max(1, length // 6)):
        add(
            '<text class="tick" x="%.1f" y="%.1f" text-anchor="middle">%d</text>'
            % (px(step), box["y1"] + 15, step)
        )
    add(
        '<line class="ax" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>'
        % (box["x0"], box["y1"], box["x1"], box["y1"])
    )
    add(
        '<text class="alab" x="%.1f" y="%.1f" text-anchor="middle">week</text>'
        % ((box["x0"] + box["x1"]) / 2, height - 5)
    )
    add(
        '<text class="alab" transform="translate(11,%.1f) rotate(-90)" '
        'text-anchor="middle">bankroll</text>' % ((box["y0"] + box["y1"]) / 2)
    )

    # Label ends are nudged apart so lines finishing close together stay legible.
    ends = sorted(
        ((label, path, py(path[-1])) for label, path in series), key=lambda row: row[2]
    )
    placed = []
    for label, path, y in ends:
        while placed and y - placed[-1] < 13:
            y = placed[-1] + 13
        placed.append(y)

    order = {label: index for index, (label, _) in enumerate(series)}
    for label, path, y in ends:
        index = order[label]
        colour = "var(--s%d)" % (index + 1)
        points = " ".join(
            "%.1f,%.1f" % (px(step), py(value)) for step, value in enumerate(path)
        )
        add(
            '<polyline class="ln" stroke="%s" points="%s"><title>%s: ends at '
            "%.3f</title></polyline>" % (colour, points, _escape(label), path[-1])
        )
        add(
            '<text class="dl" fill="%s" x="%.1f" y="%.1f">%s</text>'
            % (colour, box["x1"] + 7, y + 4, _escape(label))
        )

    add("</svg>")
    add('<div class="viz-legend">')
    for index, (label, path) in enumerate(series):
        add(
            '<span><i style="background:var(--s%d)"></i>%s &middot; %.3f</span>'
            % (index + 1, _escape(label), path[-1])
        )
    add("</div></div>")
    return "\n".join(parts)


def curve_table(curves):
    """The table view that keeps the charts from depending on colour."""
    parts = ['<div class="scroll"><table>']
    parts.append(
        "<tr><th>Series</th><th>Bucket</th><th>N</th><th>Forecast</th>"
        "<th>Observed</th><th>95% interval</th></tr>"
    )
    for label, curve in curves:
        for item in curve:
            parts.append(
                "<tr><td>%s</td><td>%.0f-%.0f%%</td><td>%d</td><td>%.1f%%</td>"
                "<td>%.1f%%</td><td>%.0f-%.0f%%</td></tr>"
                % (
                    _escape(label),
                    100 * item.low,
                    100 * item.high,
                    item.count,
                    100 * item.predicted,
                    100 * item.actual,
                    100 * item.interval_low,
                    100 * item.interval_high,
                )
            )
    parts.append("</table></div>")
    return "\n".join(parts)
