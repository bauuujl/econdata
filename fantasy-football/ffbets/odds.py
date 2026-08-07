"""Sportsbook odds math: conversions, de-vigging, expected value, Kelly.

Everything here is pure arithmetic with no dependencies, which makes it the
easiest part of the tool to unit test -- and the part most worth getting
exactly right, since a sign error turns a good bet into a bad one.
"""

from __future__ import annotations


def american_to_decimal(american):
    """-150 -> 1.667, +130 -> 2.30 (decimal odds include the stake)."""
    american = float(american)
    if american == 0:
        raise ValueError("American odds of 0 are not a valid price")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal):
    """Inverse of american_to_decimal, rounded to the nearest whole cent."""
    decimal = float(decimal)
    if decimal <= 1.0:
        raise ValueError("Decimal odds must exceed 1.0")
    if decimal >= 2.0:
        return round((decimal - 1.0) * 100.0)
    return round(-100.0 / (decimal - 1.0))


def implied_probability(american):
    """Break-even win rate baked into a price, vig included."""
    return 1.0 / american_to_decimal(american)


def probability_to_american(probability):
    """The fair (no-vig) price for a given win probability."""
    if not 0.0 < probability < 1.0:
        raise ValueError("Probability must be strictly between 0 and 1")
    return decimal_to_american(1.0 / probability)


def parse_odds(value):
    """Accept '+130', '-150', 130, or decimal odds like '2.30'.

    Decimal odds are recognised by a '.' in the string with no leading sign,
    which is how books actually distinguish the two formats in practice.
    """
    if isinstance(value, (int, float)):
        number = float(value)
        return american_to_decimal(number) if abs(number) >= 100 else float(number)

    text = str(value).strip()
    if not text:
        raise ValueError("Empty odds value")
    if text[0] in "+-":
        return american_to_decimal(float(text))
    if "." in text:
        decimal = float(text)
        if decimal <= 1.0:
            raise ValueError("Decimal odds must exceed 1.0: %s" % text)
        return decimal
    return american_to_decimal(float(text))


def remove_vig(prices):
    """Strip the book's margin from a set of prices on the same market.

    Takes decimal odds and returns fair probabilities that sum to 1, scaling
    the raw implied probabilities down proportionally (the "multiplicative"
    method).  It slightly over-taxes longshots compared with Shin's method,
    but it needs no solver and is the standard first approximation.
    """
    raw = [1.0 / price for price in prices]
    overround = sum(raw)
    if overround <= 0:
        raise ValueError("Cannot de-vig non-positive probabilities")
    return [probability / overround for probability in raw]


def hold_percentage(prices):
    """The book's theoretical margin on a market, as a percentage."""
    return (sum(1.0 / price for price in prices) - 1.0) * 100.0


def expected_value(probability, price):
    """Profit per $1 staked at decimal `price` given true `probability`."""
    return probability * (price - 1.0) - (1.0 - probability)


def kelly_fraction(probability, price, multiplier=1.0, cap=0.25):
    """Fraction of bankroll to stake, per the Kelly criterion.

    `multiplier` scales the bet down (0.25 for quarter-Kelly, the usual
    choice when your probabilities are estimates rather than truth), and
    `cap` bounds any single stake.  Never returns a negative number: a
    negative Kelly just means "no bet".
    """
    edge = expected_value(probability, price)
    if edge <= 0:
        return 0.0
    stake = edge / (price - 1.0)
    return min(stake * multiplier, cap)
