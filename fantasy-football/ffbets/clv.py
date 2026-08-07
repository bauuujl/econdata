"""Closing line value.

Win/loss results converge slowly. Sixty matchups of coin-flip-ish outcomes
cannot distinguish a 3% edge from luck -- the standard error on a 55% hit
rate over 60 bets is about 6 points. Closing line value converges much
faster, because it grades the *price* rather than the *result*: every bet
contributes a measurement instead of one bit.

The logic is that the closing line is the market's best estimate, having
absorbed all the late information and the sharpest money. If you repeatedly
take a price better than the close, you are repeatedly buying something for
less than its best available valuation, and the results follow eventually.

Three numbers are reported:

  CLV %          how much better your price was than the close, in price
                 terms. +2% means you took 2% more return than the closing
                 price offered.
  beat-close %   the share of bets taken at a better price than the close.
                 Above 50% consistently is the signal.
  EV at close    expected value computed against the *de-vigged closing
                 probability* rather than against the result. This is the
                 money-weighted version, and the one to lead with.

A caution specific to this tool: private fantasy leagues have no betting
market, so there is no real closing line to grade against. Everything here
works the moment you supply real prices, and `synthetic_market` exists to
exercise and validate the pipeline in the meantime -- not to prove an edge.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .devig import remove_vig
from .odds import decimal_to_american


@dataclass
class ClvRecord:
    """One bet, priced against the line that closed."""

    week: int
    team: str
    opponent: str
    model_probability: float
    bet_price: float  # decimal odds taken
    close_price: float = None  # decimal odds at close
    close_fair_probability: float = None  # de-vigged closing probability
    bet_fair_probability: float = None  # de-vigged probability at the bet
    won: bool = None

    @property
    def clv(self):
        """Price improvement over the close, as a fraction."""
        if not self.close_price:
            return None
        return self.bet_price / self.close_price - 1.0

    @property
    def beat_close(self):
        if not self.close_price:
            return None
        return self.bet_price > self.close_price

    @property
    def probability_clv(self):
        """CLV in probability points, once vig is out of both sides."""
        if self.close_fair_probability is None or self.bet_fair_probability is None:
            return None
        return self.close_fair_probability - self.bet_fair_probability

    @property
    def ev_at_close(self):
        """Expected profit per $1, judged by the closing line's opinion."""
        if self.close_fair_probability is None:
            return None
        return (
            self.close_fair_probability * (self.bet_price - 1.0)
            - (1.0 - self.close_fair_probability)
        )

    @property
    def realised(self):
        """Actual profit per $1 staked."""
        if self.won is None:
            return None
        return (self.bet_price - 1.0) if self.won else -1.0


@dataclass
class ClvReport:
    records: list = field(default_factory=list)

    @property
    def graded(self):
        return [record for record in self.records if record.close_price]

    @property
    def count(self):
        return len(self.graded)

    @property
    def mean_clv(self):
        graded = self.graded
        return sum(r.clv for r in graded) / len(graded) if graded else 0.0

    @property
    def beat_close_rate(self):
        graded = self.graded
        return sum(1 for r in graded if r.beat_close) / len(graded) if graded else 0.0

    @property
    def mean_ev_at_close(self):
        values = [r.ev_at_close for r in self.graded if r.ev_at_close is not None]
        return sum(values) / len(values) if values else 0.0

    @property
    def mean_probability_clv(self):
        values = [r.probability_clv for r in self.graded if r.probability_clv is not None]
        return sum(values) / len(values) if values else 0.0

    @property
    def realised_roi(self):
        values = [r.realised for r in self.graded if r.realised is not None]
        return sum(values) / len(values) if values else 0.0

    @property
    def beat_close_error(self):
        """Standard error on the beat-close rate, for reading the sample."""
        if self.count <= 0:
            return 0.0
        rate = self.beat_close_rate
        return math.sqrt(max(rate * (1 - rate), 0.0) / self.count)

    @property
    def clv_error(self):
        """Standard error on mean CLV."""
        graded = self.graded
        if len(graded) < 2:
            return 0.0
        values = [r.clv for r in graded]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance / len(values))

    @property
    def verdict(self):
        """Whether the CLV signal clears its own noise."""
        if self.count < 20:
            return "too few graded bets to read"
        edge = self.mean_clv
        error = self.clv_error
        if error <= 0:
            # Every bet moved by the same amount, so there is nothing for the
            # noise to hide: judge on the sign alone.
            if abs(edge) < 1e-12:
                return "CLV is exactly zero -- the bets closed where they opened"
            return "%s CLV of %+.2f%% on every bet, with no variance" % (
                "positive" if edge > 0 else "negative",
                100 * edge,
            )
        t_statistic = edge / error
        if t_statistic > 2:
            return "positive CLV, clear of the noise (t = %.1f)" % t_statistic
        if t_statistic > 0:
            return "positive CLV, but inside the noise (t = %.1f)" % t_statistic
        return "negative CLV (t = %.1f) -- the close disagrees with the model" % t_statistic


def build_records(bets, lines, devig_method="shin"):
    """Attach opening and closing prices to a week's bets.

    `lines` is {week: {team: {"open": price, "close": price}}} in decimal
    odds. Both sides of a matchup are de-vigged together where available, so
    the probabilities compared are free of the house margin on each end.
    """
    records = []
    for bet in bets:
        week_lines = lines.get(bet["week"]) or lines.get(str(bet["week"])) or {}
        own = week_lines.get(bet["team"])
        other = week_lines.get(bet["opponent"])
        if not own:
            continue

        record = ClvRecord(
            week=bet["week"],
            team=bet["team"],
            opponent=bet["opponent"],
            model_probability=bet["model_probability"],
            bet_price=own.get("open") or own.get("close"),
            close_price=own.get("close"),
            won=bet.get("won"),
        )

        if other and own.get("close") and other.get("close"):
            fair = remove_vig([own["close"], other["close"]], devig_method)
            record.close_fair_probability = fair[0]
        elif own.get("close"):
            record.close_fair_probability = 1.0 / own["close"]

        if other and own.get("open") and other.get("open"):
            fair = remove_vig([own["open"], other["open"]], devig_method)
            record.bet_fair_probability = fair[0]
        elif record.bet_price:
            record.bet_fair_probability = 1.0 / record.bet_price

        records.append(record)
    return ClvReport(records=records)


def load_lines(payload):
    """Normalise a user-supplied lines file into decimal odds.

    Accepts {week: {team: price}} for close-only, or
    {week: {team: {"open": price, "close": price}}} for both.
    """
    from .odds import parse_odds

    lines = {}
    for week, teams in (payload or {}).items():
        try:
            week_key = int(week)
        except (TypeError, ValueError):
            continue
        entry = {}
        for team, value in (teams or {}).items():
            if isinstance(value, dict):
                quote = {}
                if value.get("open") is not None:
                    quote["open"] = parse_odds(value["open"])
                if value.get("close") is not None:
                    quote["close"] = parse_odds(value["close"])
                if quote:
                    entry[team] = quote
            else:
                entry[team] = {"close": parse_odds(value)}
        if entry:
            lines[week_key] = entry
    return lines


def synthetic_market(
    matchups, seed=0, vig=0.045, close_noise=0.02, open_noise=0.06, devig_method="shin"
):
    """Invent an opening and closing market, to exercise the CLV pipeline.

    Private fantasy leagues have no sportsbook, so there is no real closing
    line for a past season. This builds one with a documented process:

      * the "sharp" probability is the model's own calibrated probability,
      * the close is that probability plus small noise,
      * the open is the same probability plus larger noise,
      * both are then loaded with `vig` and quoted as prices.

    Because the market is centred on the model, mean CLV here should come out
    near zero -- which is the correct answer for a model that is no better
    than the market it is being graded against, and the right way to confirm
    the metric is not manufacturing an edge. The value of running it is that
    it proves the plumbing, not that it proves a profit.
    """
    generator = random.Random(seed)
    lines = {}

    for matchup in matchups:
        week = matchup["week"]
        entry = lines.setdefault(week, {})
        anchor = min(max(matchup["fair_probability"], 0.02), 0.98)

        for team, probability in (
            (matchup["home"], anchor),
            (matchup["away"], 1.0 - anchor),
        ):
            quotes = {}
            for label, noise in (("open", open_noise), ("close", close_noise)):
                shifted = probability + generator.gauss(0.0, noise)
                # Clamp the *quoted* probability, not the fair one: loading a
                # near-certainty with vig would otherwise push the implied
                # probability past 1 and produce decimal odds below evens,
                # which is not a price a book can offer.
                implied = min(max(shifted * (1.0 + vig), 0.02), 0.95)
                quotes[label] = 1.0 / implied
            entry[team] = quotes

    return lines


def format_price(decimal_price):
    return "%+d" % decimal_to_american(decimal_price) if decimal_price else "--"
