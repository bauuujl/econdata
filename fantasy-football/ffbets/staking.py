"""Stake sizing and bankroll simulation.

Kelly is optimal only if the probability you feed it is correct. It is not,
and Kelly's penalty for overestimating an edge is asymmetric: betting twice
the Kelly fraction has zero expected log growth, and anything beyond that is
negative. Fractional Kelly is the usual guard, but a quarter of a badly
miscalibrated number is still a badly sized bet.

Two further guards are implemented here:

  shrinkage  blend the model's probability toward the market's de-vigged
             probability before sizing. The market is a strong forecast that
             has already absorbed information the model has not, so treating
             it as a prior and the model as evidence is the honest posture.
             Shrinkage of 0.3 means "the market gets 30% of the vote".
  caps       a ceiling on any single bet, and on total exposure in a week.
             Kelly with correlated bets -- and fantasy matchups in one week
             share players, so they are correlated -- will happily stake more
             in aggregate than any of the individual fractions suggest.

Bankroll paths are simulated week by week, with every bet in a week sized off
the bankroll at the start of that week and settled together, because that is
how the money actually moves.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class StakingPolicy:
    """One way of turning a probability and a price into a stake."""

    name: str
    kelly: float = 0.25  # fraction of full Kelly
    shrinkage: float = 0.0  # weight given to the market probability
    cap: float = 1.0  # ceiling on a single bet, as a fraction of bankroll
    weekly_cap: float = 1.0  # ceiling on total exposure within one week
    flat: float = None  # if set, stake this fraction instead of using Kelly
    min_edge: float = 0.0  # skip bets whose edge is thinner than this

    @property
    def description(self):
        if self.flat is not None:
            base = "flat %.1f%%" % (100 * self.flat)
        else:
            base = "%g x Kelly" % self.kelly
        extras = []
        if self.shrinkage:
            extras.append("%.0f%% market" % (100 * self.shrinkage))
        if self.cap < 1.0:
            extras.append("cap %.0f%%" % (100 * self.cap))
        if self.weekly_cap < 1.0:
            extras.append("weekly %.0f%%" % (100 * self.weekly_cap))
        return base + (", " + ", ".join(extras) if extras else "")


def blended_probability(model_probability, market_probability, shrinkage):
    """Pull the model toward the market before it is allowed to size a bet."""
    if market_probability is None or shrinkage <= 0:
        return model_probability
    weight = min(max(shrinkage, 0.0), 1.0)
    return (1.0 - weight) * model_probability + weight * market_probability


def kelly_fraction(probability, price):
    """Full-Kelly stake for a binary bet at decimal `price`."""
    if price <= 1.0:
        return 0.0
    edge = probability * (price - 1.0) - (1.0 - probability)
    if edge <= 0:
        return 0.0
    return edge / (price - 1.0)


@dataclass
class Bet:
    """One settled wager, with everything needed to re-size it after the fact."""

    week: int
    label: str
    model_probability: float
    price: float  # decimal odds actually taken
    won: bool
    market_probability: float = None  # de-vigged closing/market probability
    close_price: float = None

    def sized_probability(self, policy):
        return blended_probability(
            self.model_probability, self.market_probability, policy.shrinkage
        )

    def stake_fraction(self, policy):
        """Fraction of bankroll this policy would put on this bet."""
        probability = self.sized_probability(policy)
        edge = probability * (self.price - 1.0) - (1.0 - probability)
        if edge <= policy.min_edge:
            return 0.0
        if policy.flat is not None:
            return min(policy.flat, policy.cap)
        return min(kelly_fraction(probability, self.price) * policy.kelly, policy.cap)


@dataclass
class BankrollResult:
    policy: StakingPolicy
    path: list = field(default_factory=list)  # bankroll after each settled week
    bets_placed: int = 0
    bets_won: int = 0
    staked: float = 0.0  # total turnover, in starting-bankroll units
    profit: float = 0.0

    @property
    def final(self):
        return self.path[-1] if self.path else 1.0

    @property
    def growth(self):
        """Total return on the starting bankroll."""
        return self.final - 1.0

    @property
    def log_growth_per_week(self):
        """Geometric growth rate per week -- the quantity Kelly maximises."""
        if len(self.path) < 2 or self.final <= 0:
            return 0.0
        return math.log(self.final) / (len(self.path) - 1)

    @property
    def max_drawdown(self):
        """Deepest peak-to-trough fall along the path, as a fraction."""
        peak = self.path[0] if self.path else 1.0
        worst = 0.0
        for value in self.path:
            peak = max(peak, value)
            if peak > 0:
                worst = max(worst, (peak - value) / peak)
        return worst

    @property
    def roi(self):
        """Profit per unit staked. Immune to how big the bets were."""
        return self.profit / self.staked if self.staked > 0 else 0.0

    @property
    def hit_rate(self):
        return self.bets_won / self.bets_placed if self.bets_placed else 0.0

    @property
    def risk_adjusted(self):
        """Growth per unit of drawdown endured -- a crude Calmar ratio."""
        drawdown = self.max_drawdown
        return self.growth / drawdown if drawdown > 1e-9 else float("inf")


def simulate_bankroll(bets, policy, start=1.0, ruin_threshold=0.01):
    """Walk a bankroll through the bets, settling one week at a time.

    Stakes within a week are sized off the bankroll as it stood at the start
    of the week and scaled down together if they breach the weekly cap, which
    is what stops a week of correlated bets from quietly becoming one huge
    bet.
    """
    weeks = {}
    for bet in bets:
        weeks.setdefault(bet.week, []).append(bet)

    bankroll = start
    result = BankrollResult(policy=policy, path=[bankroll])

    for week in sorted(weeks):
        fractions = [(bet, bet.stake_fraction(policy)) for bet in weeks[week]]
        fractions = [(bet, fraction) for bet, fraction in fractions if fraction > 0]
        if not fractions:
            result.path.append(bankroll)
            continue

        total = sum(fraction for _, fraction in fractions)
        if total > policy.weekly_cap:
            scale = policy.weekly_cap / total
            fractions = [(bet, fraction * scale) for bet, fraction in fractions]

        change = 0.0
        for bet, fraction in fractions:
            stake = bankroll * fraction
            result.bets_placed += 1
            result.staked += stake
            if bet.won:
                result.bets_won += 1
                change += stake * (bet.price - 1.0)
            else:
                change -= stake

        bankroll += change
        result.profit += change
        if bankroll < ruin_threshold * start:
            # Below this the account is effectively dead; keep the path flat
            # rather than letting fractional betting limp on forever.
            bankroll = max(bankroll, 0.0)
            result.path.append(bankroll)
            break
        result.path.append(bankroll)

    return result


def default_policies(kelly=0.25):
    """A ladder from reckless to conservative, for side-by-side comparison."""
    return [
        StakingPolicy("Full Kelly", kelly=1.0),
        StakingPolicy("Half Kelly", kelly=0.5),
        StakingPolicy("Quarter Kelly", kelly=kelly),
        StakingPolicy("Quarter + 30% market", kelly=kelly, shrinkage=0.30),
        StakingPolicy(
            "Quarter + 30% market + caps",
            kelly=kelly,
            shrinkage=0.30,
            cap=0.05,
            weekly_cap=0.15,
        ),
        StakingPolicy("Flat 2%", flat=0.02),
    ]


def compare_policies(bets, policies=None, start=1.0):
    return [simulate_bankroll(bets, policy, start) for policy in (policies or default_policies())]


# ----------------------------------------------------------------------
# controlled stress test
# ----------------------------------------------------------------------
def miscalibrate(probability, overconfidence):
    """Stretch a probability away from 0.5 in log-odds space.

    `overconfidence` of 1.0 leaves it alone; 1.6 turns a true 60% into a
    stated 68%, which is roughly the distortion the Platt fits keep finding
    in this model.
    """
    probability = min(max(probability, 1e-6), 1 - 1e-6)
    odds = math.log(probability / (1 - probability)) * overconfidence
    return 1.0 / (1.0 + math.exp(-odds))


@dataclass
class StressResult:
    policy: StakingPolicy
    overconfidence: float
    median_final: float
    p05_final: float
    p95_final: float
    mean_max_drawdown: float
    ruin_rate: float
    mean_bets: float

    @property
    def median_growth(self):
        return self.median_final - 1.0


def stress_test(
    policies=None,
    overconfidence=1.6,
    seasons=400,
    weeks=14,
    bets_per_week=5,
    vig=0.045,
    seed=0,
    ruin_level=0.5,
):
    """Measure what miscalibration costs each staking rule.

    A synthetic *market* cannot settle this question -- whichever way you
    build one, you have already decided whether the model has an edge, and
    the bankroll comparison just reports your assumption back to you. A
    controlled experiment can, because the truth is specified rather than
    inferred:

      * a true win probability is drawn for each matchup,
      * the market prices that truth exactly, plus vig,
      * the bettor sees an *overconfident* version of the truth and bets
        whenever that distorted number shows value,
      * the result is drawn from the true probability.

    So the bettor has no edge at all -- every apparent edge is the
    miscalibration talking. Any policy that still grows is being paid by
    luck, and the spread across simulated seasons says how much.
    """
    policies = policies or default_policies()
    generator = random.Random(seed)
    outcomes = {policy.name: [] for policy in policies}
    drawdowns = {policy.name: [] for policy in policies}
    counts = {policy.name: [] for policy in policies}

    for _ in range(seasons):
        bets_by_policy = {policy.name: [] for policy in policies}

        for week in range(1, weeks + 1):
            for _ in range(bets_per_week):
                # Realistic spread of matchup strengths: mostly close games,
                # occasionally lopsided ones.
                true_probability = min(max(generator.gauss(0.5, 0.13), 0.05), 0.95)
                stated = miscalibrate(true_probability, overconfidence)
                price = 1.0 / (true_probability * (1.0 + vig))
                won = generator.random() < true_probability

                for policy in policies:
                    bets_by_policy[policy.name].append(
                        Bet(
                            week=week,
                            label="stress",
                            model_probability=stated,
                            price=price,
                            won=won,
                            market_probability=true_probability,
                        )
                    )

        for policy in policies:
            run = simulate_bankroll(bets_by_policy[policy.name], policy)
            outcomes[policy.name].append(run.final)
            drawdowns[policy.name].append(run.max_drawdown)
            counts[policy.name].append(run.bets_placed)

    def percentile(values, fraction):
        if not values:
            return 0.0
        ordered = sorted(values)
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    results = []
    for policy in policies:
        finals = outcomes[policy.name]
        results.append(
            StressResult(
                policy=policy,
                overconfidence=overconfidence,
                median_final=percentile(finals, 0.5),
                p05_final=percentile(finals, 0.05),
                p95_final=percentile(finals, 0.95),
                mean_max_drawdown=sum(drawdowns[policy.name]) / len(finals),
                ruin_rate=sum(1 for value in finals if value < ruin_level) / len(finals),
                mean_bets=sum(counts[policy.name]) / len(finals),
            )
        )
    return results


def render_stress(results_by_level):
    """Text table of the stress test across overconfidence levels."""
    lines = []
    out = lines.append
    rule = "-" * 78

    out("=" * 78)
    out("  STAKING STRESS TEST")
    out("=" * 78)
    out("  The bettor has NO real edge. Every apparent edge is miscalibration:")
    out("  the market prices the truth, the bettor sees a stretched version of")
    out("  it and bets whenever the distortion says there is value.")
    out("")

    for overconfidence, results in results_by_level:
        example = miscalibrate(0.60, overconfidence)
        out(
            "OVERCONFIDENCE %.2fx   (a true 60%% is stated as %.0f%%)"
            % (overconfidence, 100 * example)
        )
        out(rule)
        out(
            "  %-28s %9s %9s %9s %8s"
            % ("POLICY", "MEDIAN", "5th PCTL", "MEAN DD", "RUIN")
        )
        for item in results:
            out(
                "  %-28s %9.3f %9.3f %8.1f%% %7.1f%%"
                % (
                    item.policy.name,
                    item.median_final,
                    item.p05_final,
                    100 * item.mean_max_drawdown,
                    100 * item.ruin_rate,
                )
            )
        out("")

    out(rule)
    out("  MEDIAN is the typical season's ending bankroll from 1.0; 5th PCTL is")
    out("  the bad-luck case; RUIN is the share of seasons ending below half the")
    out("  starting bankroll. A policy whose median is below 1.0 loses money by")
    out("  construction -- the only question is how fast.")
    return "\n".join(lines)
