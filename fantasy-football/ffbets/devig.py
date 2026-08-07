"""Removing the bookmaker's margin from a set of prices.

A two-way market priced -110/-110 implies probabilities summing to 1.048.
That extra 4.8% is the house's margin, and how you take it back out changes
which side looks like value -- so this is a modelling choice, not a detail.

The four methods here differ in *where* they assume the margin sits:

  proportional  margin is spread in proportion to each raw probability.
                Simple and standard, but it taxes the favourite hardest in
                absolute terms and so systematically overstates favourites'
                fair probability. The default almost everywhere, and the
                reason this module exists.
  additive      margin is split evenly across outcomes in probability terms,
                which taxes longshots hardest. The opposite bias.
  power         each raw probability is raised to a common exponent. Sits
                between the two and has no free parameter beyond the solve.
  shin          assumes the margin exists because the book is protecting
                itself against insider money, and backs out the implied
                insider fraction. Longshot-bias-aware and generally the
                best-behaved on two-way markets, which is what a fantasy
                head-to-head is.

On a lopsided market the methods disagree by several points of probability
on the favourite, which is exactly the range where a bet is marginal.
"""

from __future__ import annotations

import math

METHODS = ("proportional", "additive", "power", "shin")

_TOLERANCE = 1e-12
_MAX_ITERATIONS = 200


def _raw_probabilities(prices):
    """Implied probabilities straight off the price, margin included."""
    if len(prices) < 2:
        raise ValueError("a market needs at least two prices")
    raw = []
    for price in prices:
        price = float(price)
        if price <= 1.0:
            raise ValueError("decimal odds must exceed 1.0, got %r" % price)
        raw.append(1.0 / price)
    return raw


def overround(prices):
    """Sum of raw implied probabilities. 1.05 means a 5% overround."""
    return sum(_raw_probabilities(prices))


def hold_percentage(prices):
    """The book's theoretical margin on a market, as a percentage."""
    return (overround(prices) - 1.0) * 100.0


# ----------------------------------------------------------------------
# methods
# ----------------------------------------------------------------------
def _proportional(raw):
    total = sum(raw)
    return [value / total for value in raw]


def _additive(raw):
    """Subtract the margin equally, then repair any negative probability.

    On a heavily lopsided market an even split can drive a longshot below
    zero; when that happens the offender is pinned just above zero and the
    remaining margin is re-split across the outcomes that can absorb it.
    """
    excess = sum(raw) - 1.0
    count = len(raw)
    fair = [value - excess / count for value in raw]

    while any(value <= 0.0 for value in fair) and count > 1:
        floor = 1e-6
        pinned = [index for index, value in enumerate(fair) if value <= floor]
        remaining = [index for index in range(len(fair)) if index not in pinned]
        if not remaining:
            break
        for index in pinned:
            fair[index] = floor
        deficit = sum(fair) - 1.0
        for index in remaining:
            fair[index] -= deficit / len(remaining)
        count = len(remaining)

    return fair


def _solve_power(raw):
    """Find k > 0 with sum(q_i ** k) == 1.

    Every q_i is below 1, so the sum falls monotonically as k rises, and a
    plain bisection is both sufficient and unconditionally stable.
    """
    low, high = 1.0, 1.0
    while sum(value ** high for value in raw) > 1.0:
        high *= 2.0
        if high > 1e6:
            raise ValueError("power de-vig failed to bracket a solution")

    for _ in range(_MAX_ITERATIONS):
        middle = 0.5 * (low + high)
        if sum(value ** middle for value in raw) > 1.0:
            low = middle
        else:
            high = middle
        if high - low < _TOLERANCE:
            break
    return 0.5 * (low + high)


def _power(raw):
    exponent = _solve_power(raw)
    fair = [value ** exponent for value in raw]
    total = sum(fair)
    return [value / total for value in fair]


def _shin_probabilities(raw, z, total):
    """Shin's fair probabilities for a given insider fraction z."""
    if z <= 0.0:
        root = math.sqrt(total)
        return [value / root for value in raw]
    fair = []
    for value in raw:
        inner = z * z + 4.0 * (1.0 - z) * value * value / total
        fair.append((math.sqrt(inner) - z) / (2.0 * (1.0 - z)))
    return fair


def _shin(raw):
    """Back out the insider fraction z that makes the fair book sum to 1."""
    total = sum(raw)
    if total <= 1.0:
        return _proportional(raw)

    low, high = 0.0, 0.99
    for _ in range(_MAX_ITERATIONS):
        middle = 0.5 * (low + high)
        if sum(_shin_probabilities(raw, middle, total)) > 1.0:
            low = middle
        else:
            high = middle
        if high - low < _TOLERANCE:
            break

    z = 0.5 * (low + high)
    fair = _shin_probabilities(raw, z, total)
    # The solve is accurate to ~1e-12; normalising absorbs the remainder so
    # callers can rely on the result summing to exactly 1.
    normaliser = sum(fair)
    return [value / normaliser for value in fair]


_DISPATCH = {
    "proportional": _proportional,
    "multiplicative": _proportional,  # the same method under its other name
    "additive": _additive,
    "power": _power,
    "shin": _shin,
}


def remove_vig(prices, method="proportional"):
    """Fair probabilities implied by `prices`, summing to 1.

    `prices` are decimal odds for every outcome in one market.
    """
    key = str(method).strip().lower()
    if key not in _DISPATCH:
        raise ValueError(
            "unknown de-vig method %r (choose from %s)" % (method, ", ".join(METHODS))
        )
    return _DISPATCH[key](_raw_probabilities(prices))


def shin_insider_fraction(prices):
    """The insider-money share Shin's model infers from a market.

    Reported for interest: a market with a high inferred insider fraction is
    one the book is pricing defensively.
    """
    raw = _raw_probabilities(prices)
    total = sum(raw)
    if total <= 1.0:
        return 0.0
    low, high = 0.0, 0.99
    for _ in range(_MAX_ITERATIONS):
        middle = 0.5 * (low + high)
        if sum(_shin_probabilities(raw, middle, total)) > 1.0:
            low = middle
        else:
            high = middle
        if high - low < _TOLERANCE:
            break
    return 0.5 * (low + high)


def compare_methods(prices):
    """Every method's fair probabilities, for inspecting the disagreement."""
    return {name: remove_vig(prices, name) for name in METHODS}
