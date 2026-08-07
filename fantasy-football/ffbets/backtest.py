"""Score the model against weeks that have already been played.

A win probability is a claim, and a completed season is the only honest way
to check it. For each finished week this replays the model using only what
was knowable beforehand -- that week's published projections and the game
logs from earlier weeks -- and then grades it four ways:

  accuracy      hit rate, Brier score, log loss.
  decomposition Murphy's split of Brier into calibration error, the
                information the model actually carries, and irreducible
                noise. This is what says whether an edge comes from knowing
                more or merely from sounding more confident.
  recalibration isotonic and Platt maps fitted out-of-sample, so the
                improvement quoted is one you would actually have got.
  money         closing line value, and bankroll paths under several staking
                rules.

Win/loss records converge slowly -- 60 matchups cannot separate a 3% edge
from luck -- which is why CLV is reported as the headline and ROI is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import calibration as calib
from .analyzer import build_lineups, team_label
from .clv import build_records, synthetic_market
from .correlations import stack_intensity
from .devig import remove_vig
from .projections import ProjectionBuilder
from .simulate import CorrelationModel, WeekSimulator, compare, summarize_team
from .staking import Bet, default_policies, simulate_bankroll


@dataclass
class Prediction:
    week: int
    home: str
    away: str
    p_home: float
    projected_home: float
    projected_away: float
    actual_home: float
    actual_away: float

    @property
    def home_won(self):
        return self.actual_home > self.actual_away

    @property
    def favorite_won(self):
        if self.actual_home == self.actual_away:
            return None
        return self.home_won if self.p_home >= 0.5 else not self.home_won

    @property
    def confidence(self):
        return max(self.p_home, 1.0 - self.p_home)


@dataclass
class BacktestResult:
    league_name: str
    season: str
    weeks: list = field(default_factory=list)
    predictions: list = field(default_factory=list)
    calibrations: dict = field(default_factory=dict)  # method -> CalibrationReport
    chosen_method: str = "none"
    clv: object = None
    bankrolls: list = field(default_factory=list)
    stack_intensity: float = 0.0
    devig_method: str = "shin"
    market_is_synthetic: bool = False
    market_mode: str = "sharp"
    bins: int = 5

    @property
    def fitted_methods(self):
        """Recalibration maps that were actually fitted, best OOS Brier first."""
        candidates = [
            report for method, report in self.calibrations.items() if method != "none"
        ]
        return sorted(
            candidates,
            key=lambda report: calib.brier_score(report.calibrated, self.outcomes),
        )

    @property
    def best_fitted(self):
        candidates = self.fitted_methods
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------
    @property
    def decided(self):
        return [p for p in self.predictions if p.favorite_won is not None]

    @property
    def raw_probabilities(self):
        return [p.p_home for p in self.decided]

    @property
    def outcomes(self):
        return [1 if p.home_won else 0 for p in self.decided]

    @property
    def groups(self):
        return [p.week for p in self.decided]

    @property
    def calibrated_probabilities(self):
        report = self.calibrations.get(self.chosen_method)
        return report.calibrated if report else self.raw_probabilities

    @property
    def hit_rate(self):
        decided = self.decided
        if not decided:
            return 0.0
        return sum(1 for p in decided if p.favorite_won) / len(decided)

    @property
    def brier(self):
        return calib.brier_score(self.raw_probabilities, self.outcomes)

    @property
    def log_loss(self):
        return calib.log_loss(self.raw_probabilities, self.outcomes)

    @property
    def skill(self):
        return 1.0 - self.brier / 0.25 if self.brier else 0.0

    @property
    def score_error(self):
        errors = []
        for prediction in self.predictions:
            errors.append(prediction.projected_home - prediction.actual_home)
            errors.append(prediction.projected_away - prediction.actual_away)
        if not errors:
            return 0.0, 0.0
        return sum(abs(e) for e in errors) / len(errors), sum(errors) / len(errors)

    def decomposition(self, calibrated=False):
        probabilities = (
            self.calibrated_probabilities if calibrated else self.raw_probabilities
        )
        return calib.murphy_decomposition(probabilities, self.outcomes, self.bins)

    def reliability(self, calibrated=False):
        probabilities = (
            self.calibrated_probabilities if calibrated else self.raw_probabilities
        )
        return calib.reliability_curve(probabilities, self.outcomes, self.bins)

    @property
    def hit_rate_error(self):
        """Standard error on the hit rate -- the reason CLV exists."""
        count = len(self.decided)
        if count <= 0:
            return 0.0
        rate = self.hit_rate
        return math.sqrt(max(rate * (1 - rate), 0.0) / count)


# ----------------------------------------------------------------------
def backtest_league(
    client,
    league_id,
    first_week=2,
    last_week=None,
    n_sims=4000,
    seed=1,
    history_weeks=8,
    projection_weight=0.80,
    correlation=None,
    ensemble=None,
    calibration_methods=("platt", "isotonic"),
    calibration_folds=5,
    devig_method="shin",
    lines=None,
    kelly=0.25,
    policies=None,
    bins=5,
    market_mode="sharp",
    verbose=False,
):
    league = client.league(league_id)
    season = str(league["season"])
    scoring = league["scoring_settings"]
    roster_positions = league.get("roster_positions") or []
    rosters = client.league_rosters(league_id)
    users = {user["user_id"]: user for user in client.league_users(league_id)}
    owner_of = {roster["roster_id"]: roster.get("owner_id") for roster in rosters}

    if last_week is None:
        last_week = (league.get("settings") or {}).get("playoff_week_start", 15) - 1

    try:
        player_db = client.players()
    except Exception:
        player_db = {}

    result = BacktestResult(
        league_name=league.get("name", "League"),
        season=season,
        devig_method=devig_method,
        bins=bins,
    )

    stack_samples = []
    for week in range(int(first_week), int(last_week) + 1):
        try:
            matchups = client.matchups(league_id, week)
        except Exception as exc:
            if verbose:
                print("  week %s unavailable (%s)" % (week, exc))
            continue
        if not matchups or not any(entry.get("points") for entry in matchups):
            continue

        builder = ProjectionBuilder(
            client,
            scoring,
            season,
            week,
            projection_weight=projection_weight,
            history_weeks=history_weeks,
            player_db=player_db,
            ensemble=ensemble,
            verbose=False,
        )
        built, _ = build_lineups(
            matchups, rosters, builder, roster_positions, mode="declared"
        )

        simulator = WeekSimulator(
            correlation=correlation or CorrelationModel(),
            n_sims=n_sims,
            seed=seed + week,
        )
        ordered = [(roster_id, built[roster_id]["starters"]) for roster_id in built]
        stack_samples.append(stack_intensity(ordered))
        totals = simulator.run(ordered)

        actual = {entry["roster_id"]: (entry.get("points") or 0.0) for entry in matchups}
        teams = {}
        for roster_id, payload in built.items():
            user = users.get(owner_of.get(roster_id))
            teams[roster_id] = summarize_team(
                roster_id,
                team_label(user, roster_id),
                (user or {}).get("display_name", "?"),
                payload["starters"],
                totals[roster_id],
            )

        grouped = {}
        for entry in matchups:
            grouped.setdefault(entry.get("matchup_id"), []).append(entry["roster_id"])

        counted = 0
        for roster_ids in grouped.values():
            if len(roster_ids) != 2:
                continue
            home, away = teams[roster_ids[0]], teams[roster_ids[1]]
            head_to_head = compare(None, home, away)
            result.predictions.append(
                Prediction(
                    week=week,
                    home=home.name,
                    away=away.name,
                    p_home=head_to_head.p_home,
                    projected_home=home.mean,
                    projected_away=away.mean,
                    actual_home=actual.get(home.roster_id, 0.0),
                    actual_away=actual.get(away.roster_id, 0.0),
                )
            )
            counted += 1

        if counted:
            result.weeks.append(week)
        if verbose:
            print("  week %s: %d matchups graded" % (week, counted))

    result.stack_intensity = (
        sum(stack_samples) / len(stack_samples) if stack_samples else 0.0
    )

    _fit_calibrations(result, calibration_methods, calibration_folds, seed)
    _grade_money(result, lines, devig_method, kelly, policies, market_mode)
    return result


def _fit_calibrations(result, methods, folds, seed):
    """Fit every requested recalibration map out-of-sample and keep the best."""
    if not result.decided:
        return
    probabilities = result.raw_probabilities
    outcomes = result.outcomes
    groups = result.groups

    result.calibrations["none"] = calib.CalibrationReport(
        method="none",
        raw=list(probabilities),
        calibrated=list(probabilities),
        outcomes=list(outcomes),
        fitted=calib.IdentityCalibrator(),
        folds=0,
        description="no adjustment",
    )

    for method in methods:
        result.calibrations[method] = calib.cross_validated_calibration(
            probabilities, outcomes, groups, method=method, folds=folds, seed=seed
        )

    # Pick on out-of-sample Brier. A map that does not beat leaving the
    # probabilities alone does not get used.
    best = min(
        result.calibrations.values(),
        key=lambda report: calib.brier_score(report.calibrated, outcomes),
    )
    result.chosen_method = best.method


def _grade_money(result, lines, devig_method, kelly, policies, market_mode="sharp"):
    """Attach a market, then measure CLV and simulate bankrolls.

    With no real prices, the market is simulated -- and *which* probability
    it is centred on decides what the experiment actually tests:

      model  the market agrees with the model. CLV then comes out near zero,
             which confirms the plumbing works but tells you nothing about
             staking, because the model has a real edge by construction and
             Full Kelly always wins.
      sharp  the market is centred on the model's *recalibrated* probability
             while the bettor stakes off the *raw* one. That is the realistic
             adversarial case -- a bettor who is overconfident relative to a
             sharper market -- and it is the only setting in which the
             shrinkage and cap variants can show what they are for.
    """
    if not result.decided:
        return

    decided = result.decided
    result.market_mode = market_mode

    # In 'sharp' mode the market is anchored on the *fitted* recalibration
    # rather than the selected one. The Platt fit measures how overconfident
    # the raw model is on this very data, so pricing the market at the
    # corrected probability and betting the raw one reproduces exactly the
    # miscalibration that was actually observed. It is a stress test of the
    # staking rules, not a claim that the corrected numbers are more accurate
    # -- the out-of-sample table above is what speaks to that.
    fitted = result.best_fitted
    if market_mode == "sharp" and fitted is not None and fitted.fitted is not None:
        anchor = [fitted.fitted.predict(p) for p in result.raw_probabilities]
        betting_probabilities = result.raw_probabilities
    else:
        anchor = result.calibrated_probabilities
        betting_probabilities = result.calibrated_probabilities
    calibrated = anchor

    if lines is None:
        result.market_is_synthetic = True
        # The market is always anchored on the best available estimate.
        lines = synthetic_market(
            [
                {
                    "week": prediction.week,
                    "home": prediction.home,
                    "away": prediction.away,
                    "fair_probability": probability,
                }
                for prediction, probability in zip(decided, calibrated)
            ],
            seed=17,
            devig_method=devig_method,
        )
    else:
        betting_probabilities = calibrated

    bet_rows = []
    bets = []
    for prediction, probability in zip(decided, betting_probabilities):
        week_lines = lines.get(prediction.week) or lines.get(str(prediction.week)) or {}
        home_quote = week_lines.get(prediction.home)
        away_quote = week_lines.get(prediction.away)
        if not home_quote or not away_quote:
            continue

        for team, opponent, model_probability, quote, other, won in (
            (
                prediction.home,
                prediction.away,
                probability,
                home_quote,
                away_quote,
                prediction.home_won,
            ),
            (
                prediction.away,
                prediction.home,
                1.0 - probability,
                away_quote,
                home_quote,
                not prediction.home_won,
            ),
        ):
            price = quote.get("open") or quote.get("close")
            if not price:
                continue

            # The market probability used for shrinkage must be the one
            # available when the bet is struck. Using the closing line here
            # would leak information from after the decision.
            other_open = other.get("open") or other.get("close")
            if other_open:
                market_probability = remove_vig([price, other_open], devig_method)[0]
            else:
                market_probability = 1.0 / price

            edge = model_probability * (price - 1.0) - (1.0 - model_probability)
            if edge <= 0:
                continue

            bet_rows.append(
                {
                    "week": prediction.week,
                    "team": team,
                    "opponent": opponent,
                    "model_probability": model_probability,
                    "won": won,
                }
            )
            bets.append(
                Bet(
                    week=prediction.week,
                    label="%s wk%d" % (team, prediction.week),
                    model_probability=model_probability,
                    price=price,
                    won=won,
                    market_probability=market_probability,
                    close_price=quote.get("close"),
                )
            )

    result.clv = build_records(bet_rows, lines, devig_method)
    result.bankrolls = [
        simulate_bankroll(bets, policy) for policy in (policies or default_policies(kelly))
    ]


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------
def render_backtest(result, top=8):
    lines = []
    out = lines.append
    rule = "-" * 78

    out("=" * 78)
    out("  BACKTEST  |  %s  |  %s season" % (result.league_name, result.season))
    out("=" * 78)

    decided = result.decided
    if not decided:
        out("  No completed matchups to grade.")
        return "\n".join(lines)

    weeks = result.weeks
    out(
        "  weeks %s-%s  |  %d matchups graded  |  %.1f stacked starter pairs per roster"
        % (min(weeks), max(weeks), len(decided), result.stack_intensity)
    )

    out("")
    out("ACCURACY")
    out(rule)
    out(
        "  Favourite won            %.1f%% of the time  (+/- %.1f)"
        % (100 * result.hit_rate, 100 * result.hit_rate_error)
    )
    out("  Brier score              %.4f   (0.25 = coin flip, lower is better)" % result.brier)
    out("  Brier skill vs coinflip  %+.1f%%" % (100 * result.skill))
    out("  Log loss                 %.4f   (0.693 = coin flip)" % result.log_loss)

    mae, bias = result.score_error
    out("  Team score MAE           %.1f points   (bias %+.1f)" % (mae, bias))

    out("")
    out("WHERE THE SCORE COMES FROM  (Murphy decomposition of Brier)")
    out(rule)
    raw = result.decomposition(False)
    tuned = result.decomposition(True)
    out("  %-26s %10s %12s" % ("", "RAW", "CALIBRATED"))
    out("  %-26s %10.4f %12.4f" % ("Brier", raw.brier, tuned.brier))
    out(
        "  %-26s %10.4f %12.4f   lower is better"
        % ("  reliability (calibration)", raw.reliability, tuned.reliability)
    )
    out(
        "  %-26s %10.4f %12.4f   higher is better"
        % ("  resolution (discrimination)", raw.resolution, tuned.resolution)
    )
    out(
        "  %-26s %10.4f %12.4f   irreducible"
        % ("  uncertainty", raw.uncertainty, tuned.uncertainty)
    )
    out("")
    out(
        "  Of the %.4f achievable, discrimination earns %.4f and miscalibration"
        % (raw.uncertainty, raw.resolution)
    )
    out(
        "  gives back %.4f. The edge is %s."
        % (
            raw.reliability,
            "real information, partly wasted by overconfidence"
            if raw.resolution > raw.reliability
            else "mostly an artefact of miscalibration",
        )
    )

    best_fitted = result.best_fitted
    out("")
    out(
        "RECALIBRATION  (fitted out-of-sample, %d folds grouped by week)"
        % (best_fitted.folds if best_fitted else 0)
    )
    out(rule)
    out("  %-10s %10s %11s %11s  %s" % ("METHOD", "BRIER", "LOG LOSS", "IMPROVE", "FIT"))
    for method, report in result.calibrations.items():
        marker = "*" if method == result.chosen_method else " "
        scores = report.scores(result.bins)
        out(
            "%s %-10s %10.4f %11.4f %11s  %s"
            % (
                marker,
                method,
                scores["calibrated_brier"],
                scores["calibrated_log_loss"],
                "%+.4f" % report.improvement if method != "none" else "--",
                report.description,
            )
        )
    out("")
    out("  * selected on out-of-sample Brier and used for every number below.")

    out("")
    out("RELIABILITY CURVE  (a calibrated model has PREDICTED == ACTUAL)")
    out(rule)

    def curve_block(title, curve):
        out("  %s" % title)
        out(
            "    %-14s %6s %11s %10s %18s"
            % ("BUCKET", "N", "PREDICTED", "ACTUAL", "95% INTERVAL")
        )
        for item in curve:
            flag = "  <-- off" if not (
                item.interval_low <= item.predicted <= item.interval_high
            ) else ""
            out(
                "    %-14s %6d %10.1f%% %9.1f%%   %5.0f%% - %3.0f%%%s"
                % (
                    "%.0f-%.0f%%" % (100 * item.low, 100 * item.high),
                    item.count,
                    100 * item.predicted,
                    100 * item.actual,
                    100 * item.interval_low,
                    100 * item.interval_high,
                    flag,
                )
            )

    curve_block("RAW", result.reliability(False))
    if best_fitted is not None:
        out("")
        curve_block(
            "AFTER %s (out-of-sample)" % best_fitted.method.upper(),
            calib.reliability_curve(
                best_fitted.calibrated, result.outcomes, result.bins
            ),
        )
    out("")
    out(
        "  A bucket is flagged when its forecast falls outside the interval its"
    )
    out(
        "  own outcomes support -- with %d matchups spread over %d buckets, most"
        % (len(decided), result.bins)
    )
    out("  will not be, and that is a statement about the sample, not the model.")

    if result.clv and result.clv.count:
        out("")
        out("CLOSING LINE VALUE  (the primary metric)")
        out(rule)
        clv = result.clv
        out("  Bets graded              %d" % clv.count)
        out(
            "  Mean CLV                 %+.2f%%  (+/- %.2f)"
            % (100 * clv.mean_clv, 100 * clv.clv_error)
        )
        out(
            "  Beat the close           %.1f%%  (+/- %.1f)"
            % (100 * clv.beat_close_rate, 100 * clv.beat_close_error)
        )
        out("  Mean EV at close         %+.3f per $1 staked" % clv.mean_ev_at_close)
        out("  Mean CLV in probability  %+.2f points" % (100 * clv.mean_probability_clv))
        out("  Realised ROI             %+.3f per $1 staked" % clv.realised_roi)
        out("")
        out("  Verdict: %s." % clv.verdict)
        if result.market_is_synthetic:
            out("")
            out("  !! SIMULATED MARKET -- private fantasy leagues have no sportsbook,")
            out("     so there is no real closing line for a past season.")
            if result.market_mode == "sharp":
                out("     Mode 'sharp': the market is centred on the RECALIBRATED")
                out("     probability while the bettor stakes off the RAW one, so a")
                out("     negative CLV here is the correct answer -- it is the model")
                out("     being marked against a sharper opinion, which is exactly")
                out("     the condition the staking variants below are tested under.")
            else:
                out("     Mode 'model': the market is centred on the model itself, so")
                out("     CLV near zero is correct and proves only the plumbing.")
            out("     Supply real prices with --lines to measure genuine CLV.")

    if result.bankrolls:
        out("")
        out("STAKING  (bankroll = 1.0, settled week by week)")
        out(rule)
        out(
            "  %-28s %8s %9s %9s %7s"
            % ("POLICY", "FINAL", "GROWTH", "MAX DD", "BETS")
        )
        for run in result.bankrolls:
            out(
                "  %-28s %8.3f %8.1f%% %8.1f%% %7d"
                % (
                    run.policy.name,
                    run.final,
                    100 * run.growth,
                    100 * run.max_drawdown,
                    run.bets_placed,
                )
            )
        out("")
        best = max(result.bankrolls, key=lambda run: run.risk_adjusted)
        out("")
        out(
            "  Best growth per unit of drawdown: %s (%.1f%% growth, %.1f%% max drawdown)."
            % (best.policy.name, 100 * best.growth, 100 * best.max_drawdown)
        )
        if result.market_is_synthetic and result.market_mode == "sharp":
            full = next(
                (r for r in result.bankrolls if r.policy.name == "Full Kelly"), None
            )
            shrunk = next(
                (r for r in result.bankrolls if "market" in r.policy.name), None
            )
            if full and shrunk:
                out(
                    "  Full Kelly on miscalibrated probabilities: %.1f%% growth for a"
                    % (100 * full.growth)
                )
                out(
                    "  %.1f%% drawdown. Shrinking 30%% toward the market: %.1f%% growth"
                    % (100 * full.max_drawdown, 100 * shrunk.growth)
                )
                out(
                    "  for a %.1f%% drawdown. That is the trade the shrinkage buys."
                    % (100 * shrunk.max_drawdown)
                )

    out("")
    out(rule)
    out(
        "  %d matchups is a small sample: the hit rate alone carries a +/-%.1f point"
        % (len(decided), 100 * result.hit_rate_error)
    )
    out("  standard error, which is why CLV leads and ROI does not.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------
def render_backtest_html(result, title=None):
    """Self-contained HTML: the same numbers, plus the reliability plot."""
    import html as html_module

    from . import calibration as calib_module
    from .charts import CHART_CSS, bankroll_chart, curve_table, reliability_chart
    from .report import CSS

    e = html_module.escape
    parts = []
    add = parts.append

    heading = title or "%s - model report" % result.league_name
    add("<title>%s</title>" % e(heading))
    add("<style>%s%s</style>" % (CSS, CHART_CSS))
    add('<div class="wrap">')
    add("<h1>%s</h1>" % e(heading))

    decided = result.decided
    if not decided:
        add("<p>No completed matchups to grade.</p></div>")
        return "\n".join(parts)

    add(
        '<div class="sub">%s season &middot; weeks %d-%d &middot; %d matchups graded '
        "&middot; %.1f stacked starter pairs per roster</div>"
        % (
            e(result.season),
            min(result.weeks),
            max(result.weeks),
            len(decided),
            result.stack_intensity,
        )
    )

    raw = result.decomposition(False)
    add("<h2>Headline</h2>")
    add('<div class="scroll"><table>')
    add("<tr><th>Measure</th><th>Value</th><th>Reference</th></tr>")
    for label, value, reference in (
        ("Favourite won", "%.1f%% (+/- %.1f)" % (100 * result.hit_rate, 100 * result.hit_rate_error), "50% = coin flip"),
        ("Brier score", "%.4f" % result.brier, "0.25 = coin flip"),
        ("Brier skill", "%+.1f%%" % (100 * result.skill), "0% = no skill"),
        ("Log loss", "%.4f" % result.log_loss, "0.693 = coin flip"),
        ("Team score MAE", "%.1f pts" % result.score_error[0], "bias %+.1f" % result.score_error[1]),
    ):
        add("<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (e(label), e(value), e(reference)))
    add("</table></div>")

    add("<h2>Where the score comes from</h2>")
    tuned = result.decomposition(True)
    add('<div class="scroll"><table>')
    add("<tr><th>Term</th><th>Raw</th><th>Calibrated</th><th>Reading</th></tr>")
    for label, a, b, reading in (
        ("Brier", raw.brier, tuned.brier, "lower is better"),
        ("Reliability (calibration error)", raw.reliability, tuned.reliability, "lower is better"),
        ("Resolution (discrimination)", raw.resolution, tuned.resolution, "higher is better"),
        ("Uncertainty", raw.uncertainty, tuned.uncertainty, "irreducible"),
    ):
        add(
            "<tr><td>%s</td><td>%.4f</td><td>%.4f</td><td>%s</td></tr>"
            % (e(label), a, b, e(reading))
        )
    add("</table></div>")
    add(
        "<p class='viz-sub'>Brier = reliability &minus; resolution + uncertainty. "
        "Resolution is information the model carries; reliability is how much of "
        "it overconfidence gives back. Recalibration can move the second, never "
        "the first.</p>"
    )

    curves = [("raw", result.reliability(False))]
    best_fitted = result.best_fitted
    if best_fitted is not None:
        curves.append(
            (
                best_fitted.method,
                calib_module.reliability_curve(
                    best_fitted.calibrated, result.outcomes, result.bins
                ),
            )
        )

    add("<h2>Reliability</h2>")
    add(
        reliability_chart(
            curves,
            title="Forecast vs outcome, before and after recalibration",
            subtitle="Points below the dashed line are forecasts that were too "
            "confident. Bars are 95% intervals on the observed rate.",
        )
    )
    add(curve_table(curves))

    add("<h2>Recalibration (out-of-sample)</h2>")
    add('<div class="scroll"><table>')
    add("<tr><th>Method</th><th>Brier</th><th>Log loss</th><th>Improvement</th><th>Fit</th></tr>")
    for method, report in result.calibrations.items():
        scores = report.scores(result.bins)
        mark = " &larr; used" if method == result.chosen_method else ""
        add(
            "<tr><td>%s%s</td><td>%.4f</td><td>%.4f</td><td class='%s'>%s</td><td>%s</td></tr>"
            % (
                e(method),
                mark,
                scores["calibrated_brier"],
                scores["calibrated_log_loss"],
                "pos" if report.improvement > 0 else "neg",
                "%+.4f" % report.improvement if method != "none" else "&mdash;",
                e(report.description),
            )
        )
    add("</table></div>")

    if result.clv and result.clv.count:
        clv = result.clv
        add("<h2>Closing line value</h2>")
        add('<div class="scroll"><table>')
        add("<tr><th>Measure</th><th>Value</th></tr>")
        for label, value in (
            ("Bets graded", "%d" % clv.count),
            ("Mean CLV", "%+.2f%% (+/- %.2f)" % (100 * clv.mean_clv, 100 * clv.clv_error)),
            ("Beat the close", "%.1f%% (+/- %.1f)" % (100 * clv.beat_close_rate, 100 * clv.beat_close_error)),
            ("Mean EV at close", "%+.3f per $1" % clv.mean_ev_at_close),
            ("Realised ROI", "%+.3f per $1" % clv.realised_roi),
        ):
            add("<tr><td>%s</td><td>%s</td></tr>" % (e(label), e(value)))
        add("</table></div>")
        add("<p class='viz-sub'>%s</p>" % e(clv.verdict.capitalize() + "."))
        if result.market_is_synthetic:
            add(
                '<div class="verdict none"><strong>Simulated market.</strong> '
                "Private fantasy leagues have no sportsbook, so there is no real "
                "closing line for a past season. These prices are generated, which "
                "makes this a test of the pipeline rather than evidence of an edge. "
                "Supply real prices with <code>--lines</code> to measure genuine CLV."
                "</div>"
            )

    if result.bankrolls:
        add("<h2>Staking</h2>")
        add(
            bankroll_chart(
                [(run.policy.name, run.path) for run in result.bankrolls],
                title="Bankroll by staking policy",
                subtitle="Starting from 1.0, settled week by week. Dashed line is "
                "break even.",
            )
        )
        add('<div class="scroll"><table>')
        add(
            "<tr><th>Policy</th><th>Rule</th><th>Final</th><th>Growth</th>"
            "<th>Max drawdown</th><th>Bets</th></tr>"
        )
        for run in result.bankrolls:
            add(
                "<tr><td>%s</td><td>%s</td><td>%.3f</td><td class='%s'>%+.1f%%</td>"
                "<td>%.1f%%</td><td>%d</td></tr>"
                % (
                    e(run.policy.name),
                    e(run.policy.description),
                    run.final,
                    "pos" if run.growth > 0 else "neg",
                    100 * run.growth,
                    100 * run.max_drawdown,
                    run.bets_placed,
                )
            )
        add("</table></div>")

    add(
        "<footer>%d matchups carries a &plusmn;%.1f point standard error on the hit "
        "rate alone, which is why closing line value leads this report and ROI does "
        "not. Projections are estimates, not outcomes.</footer>"
        % (len(decided), 100 * result.hit_rate_error)
    )
    add("</div>")
    return "\n".join(parts)
