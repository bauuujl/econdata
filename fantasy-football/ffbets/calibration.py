"""Recalibrating probabilities, and decomposing how good they are.

A model can rank matchups well and still be systematically overconfident.
Ranking is *discrimination*; saying 70% and being right 70% of the time is
*calibration*. Betting pays for calibration: expected value is computed
directly from the probability, so a probability that is 10 points too high
turns a losing bet into one that looks like value.

Two recalibration maps are offered:

  Platt    a logistic fit on the log-odds of the raw probability. Two
           parameters, so it cannot overfit a small sample, but it can only
           apply a smooth monotone squeeze -- it fixes global overconfidence
           and nothing else.
  Isotonic a free monotone step function. It can correct any monotone
           distortion, including one that only affects big favourites, at
           the cost of far more freedom to fit noise.

With a season of matchups the sample is small enough that Platt is usually
the safer choice, and the cross-validated scores here are what tell you
which is actually helping rather than memorising.

Everything is fitted out-of-sample. Fitting a calibrator on all the data and
then scoring it on that same data will always look like an improvement, and
that improvement is not real.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

EPSILON = 1e-6


def _clip(probability, epsilon=EPSILON):
    return min(max(float(probability), epsilon), 1.0 - epsilon)


def _logit(probability):
    probability = _clip(probability)
    return math.log(probability / (1.0 - probability))


def _sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


# ----------------------------------------------------------------------
# scoring rules
# ----------------------------------------------------------------------
def brier_score(probabilities, outcomes):
    """Mean squared error of probabilistic forecasts. Lower is better."""
    if not probabilities:
        return 0.0
    return sum(
        (p - y) ** 2 for p, y in zip(probabilities, outcomes)
    ) / len(probabilities)


def log_loss(probabilities, outcomes):
    if not probabilities:
        return 0.0
    total = 0.0
    for probability, outcome in zip(probabilities, outcomes):
        probability = _clip(probability)
        total -= outcome * math.log(probability) + (1 - outcome) * math.log(
            1 - probability
        )
    return total / len(probabilities)


def brier_skill(probabilities, outcomes, reference=None):
    """Improvement over always forecasting the base rate."""
    if not probabilities:
        return 0.0
    if reference is None:
        reference = sum(outcomes) / len(outcomes)
    baseline = sum((reference - y) ** 2 for y in outcomes) / len(outcomes)
    if baseline <= 0:
        return 0.0
    return 1.0 - brier_score(probabilities, outcomes) / baseline


# ----------------------------------------------------------------------
# binning and the Murphy decomposition
# ----------------------------------------------------------------------
def quantile_bins(probabilities, count):
    """Bin edges putting roughly equal numbers of forecasts in each bin.

    Equal-width bins leave the extreme bins nearly empty on a small sample,
    which makes a reliability curve look far worse than it is.
    """
    ordered = sorted(probabilities)
    if not ordered:
        return [0.0, 1.0]
    edges = [0.0]
    for index in range(1, count):
        position = int(round(index * len(ordered) / count))
        position = min(max(position, 0), len(ordered) - 1)
        edge = ordered[position]
        if edge > edges[-1] + 1e-9:
            edges.append(edge)
    edges.append(1.0 + 1e-9)
    return edges


def _assign_bins(probabilities, edges):
    groups = [[] for _ in range(len(edges) - 1)]
    for index, probability in enumerate(probabilities):
        slot = len(edges) - 2
        for edge_index in range(len(edges) - 1):
            if probability < edges[edge_index + 1]:
                slot = edge_index
                break
        groups[slot].append(index)
    return groups


def wilson_interval(successes, total, z=1.96):
    """Confidence interval for an observed rate.

    Plotted as error bars on the reliability curve so that a bin holding
    seven matchups is not read as though it held seven hundred.
    """
    if total <= 0:
        return 0.0, 1.0
    rate = successes / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, centre - spread), min(1.0, centre + spread)


@dataclass
class ReliabilityBin:
    low: float
    high: float
    count: int
    predicted: float
    actual: float
    interval_low: float
    interval_high: float


def reliability_curve(probabilities, outcomes, bins=5, strategy="quantile"):
    """Predicted vs realised rate, bucketed -- the data behind the plot."""
    if not probabilities:
        return []
    if strategy == "quantile":
        edges = quantile_bins(probabilities, bins)
    else:
        edges = [index / bins for index in range(bins + 1)]
        edges[-1] = 1.0 + 1e-9

    curve = []
    for group, indices in enumerate(_assign_bins(probabilities, edges)):
        if not indices:
            continue
        predicted = sum(probabilities[i] for i in indices) / len(indices)
        successes = sum(outcomes[i] for i in indices)
        low, high = wilson_interval(successes, len(indices))
        curve.append(
            ReliabilityBin(
                low=edges[group],
                high=min(edges[group + 1], 1.0),
                count=len(indices),
                predicted=predicted,
                actual=successes / len(indices),
                interval_low=low,
                interval_high=high,
            )
        )
    return curve


@dataclass
class MurphyDecomposition:
    """Brier = reliability - resolution + uncertainty.

    reliability  calibration error. How far the realised rate sits from the
                 forecast rate within each bin. Zero is perfect; this is the
                 term recalibration attacks.
    resolution   discrimination. How far the bins' outcomes spread away from
                 the base rate. Bigger is better -- it is the information
                 the model actually carries, and recalibration cannot
                 create it.
    uncertainty  the irreducible variance of the outcome itself. Nothing a
                 model does can change it; it is the score a forecaster who
                 only knows the base rate would post.
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    binned_brier: float
    bins: int

    @property
    def discretization_gap(self):
        """Brier minus its binned reconstruction, from within-bin spread.

        The three-term identity is exact for the binned forecast, not the raw
        one. A large gap means the bins are too coarse to describe the model.
        """
        return self.brier - self.binned_brier

    @property
    def skill_from_resolution(self):
        """Share of the achievable skill that comes from discrimination."""
        if self.uncertainty <= 0:
            return 0.0
        return self.resolution / self.uncertainty

    @property
    def loss_from_reliability(self):
        """Share of achievable skill given back to miscalibration."""
        if self.uncertainty <= 0:
            return 0.0
        return self.reliability / self.uncertainty


def murphy_decomposition(probabilities, outcomes, bins=5, strategy="quantile"):
    """Split the Brier score into calibration, discrimination and noise."""
    total = len(probabilities)
    if not total:
        return MurphyDecomposition(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    base_rate = sum(outcomes) / total
    uncertainty = base_rate * (1.0 - base_rate)

    curve = reliability_curve(probabilities, outcomes, bins, strategy)
    reliability = sum(
        item.count * (item.predicted - item.actual) ** 2 for item in curve
    ) / total
    resolution = sum(
        item.count * (item.actual - base_rate) ** 2 for item in curve
    ) / total

    return MurphyDecomposition(
        brier=brier_score(probabilities, outcomes),
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        binned_brier=reliability - resolution + uncertainty,
        bins=len(curve),
    )


# ----------------------------------------------------------------------
# calibrators
# ----------------------------------------------------------------------
class Calibrator:
    """Maps a raw probability to a corrected one."""

    name = "identity"

    def fit(self, probabilities, outcomes):
        return self

    def predict(self, probability):
        return _clip(probability)

    def predict_all(self, probabilities):
        return [self.predict(p) for p in probabilities]

    def to_dict(self):
        return {"name": self.name}

    @property
    def description(self):
        return "no adjustment"


class IdentityCalibrator(Calibrator):
    pass


class PlattCalibrator(Calibrator):
    """p_calibrated = sigmoid(a * logit(p) + b), fitted by Newton's method.

    `a` below 1 means the raw model is overconfident: it pulls extreme
    forecasts back toward the middle. `b` shifts the whole curve, correcting
    a systematic lean toward one side.
    """

    name = "platt"

    def __init__(self, slope=1.0, intercept=0.0, smoothing=True):
        self.slope = slope
        self.intercept = intercept
        self.smoothing = smoothing

    def fit(self, probabilities, outcomes, iterations=100):
        features = [_logit(p) for p in probabilities]
        targets = self._targets(outcomes)

        slope, intercept = 1.0, 0.0
        for _ in range(iterations):
            # Newton step on the cross-entropy loss. Two parameters, so the
            # Hessian is 2x2 and inverts in closed form.
            g0 = g1 = h00 = h01 = h11 = 0.0
            for feature, target in zip(features, targets):
                fitted = _sigmoid(slope * feature + intercept)
                residual = fitted - target
                weight = max(fitted * (1.0 - fitted), 1e-12)
                g0 += residual * feature
                g1 += residual
                h00 += weight * feature * feature
                h01 += weight * feature
                h11 += weight

            determinant = h00 * h11 - h01 * h01
            if abs(determinant) < 1e-12:
                break
            step_slope = (h11 * g0 - h01 * g1) / determinant
            step_intercept = (h00 * g1 - h01 * g0) / determinant
            slope -= step_slope
            intercept -= step_intercept
            if abs(step_slope) < 1e-10 and abs(step_intercept) < 1e-10:
                break

        self.slope, self.intercept = slope, intercept
        return self

    def _targets(self, outcomes):
        """Platt's smoothed targets, which keep a clean split from diverging.

        Without them a sample where every forecast above 60% won would push
        the fit toward infinite slope.
        """
        if not self.smoothing:
            return [float(y) for y in outcomes]
        positives = sum(1 for y in outcomes if y > 0.5)
        negatives = len(outcomes) - positives
        high = (positives + 1.0) / (positives + 2.0)
        low = 1.0 / (negatives + 2.0)
        return [high if y > 0.5 else low for y in outcomes]

    def predict(self, probability):
        return _clip(_sigmoid(self.slope * _logit(probability) + self.intercept))

    def to_dict(self):
        return {"name": self.name, "slope": self.slope, "intercept": self.intercept}

    @property
    def description(self):
        lean = "overconfident" if self.slope < 1 else "underconfident"
        return "slope %.3f (%s), intercept %+.3f" % (self.slope, lean, self.intercept)


class IsotonicCalibrator(Calibrator):
    """A free monotone map fitted by pool-adjacent-violators.

    Stored as a set of knots and evaluated by linear interpolation, so the
    output is continuous rather than a staircase -- a staircase would map a
    whole range of raw probabilities onto one identical price.
    """

    name = "isotonic"

    def __init__(self, knots_x=None, knots_y=None):
        self.knots_x = knots_x or []
        self.knots_y = knots_y or []

    def fit(self, probabilities, outcomes):
        paired = sorted(zip(probabilities, outcomes))
        if not paired:
            return self

        values = [float(y) for _, y in paired]
        weights = [1.0] * len(values)
        counts = [1] * len(values)

        # Pool adjacent violators: repeatedly merge any pair that decreases.
        stack_values, stack_weights, stack_counts = [], [], []
        for value, weight, count in zip(values, weights, counts):
            stack_values.append(value)
            stack_weights.append(weight)
            stack_counts.append(count)
            while len(stack_values) > 1 and stack_values[-2] > stack_values[-1] + 1e-15:
                v2, w2, c2 = (
                    stack_values.pop(),
                    stack_weights.pop(),
                    stack_counts.pop(),
                )
                v1, w1, c1 = (
                    stack_values.pop(),
                    stack_weights.pop(),
                    stack_counts.pop(),
                )
                merged = w1 + w2
                stack_values.append((v1 * w1 + v2 * w2) / merged)
                stack_weights.append(merged)
                stack_counts.append(c1 + c2)

        fitted = []
        for value, count in zip(stack_values, stack_counts):
            fitted.extend([value] * count)

        xs = [x for x, _ in paired]
        # Keep only the knots where the fitted value changes, plus the ends,
        # so interpolation stays faithful without storing every point.
        knots_x, knots_y = [xs[0]], [fitted[0]]
        for index in range(1, len(xs)):
            if abs(fitted[index] - fitted[index - 1]) > 1e-12:
                knots_x.append(xs[index - 1])
                knots_y.append(fitted[index - 1])
                knots_x.append(xs[index])
                knots_y.append(fitted[index])
        knots_x.append(xs[-1])
        knots_y.append(fitted[-1])

        self.knots_x, self.knots_y = knots_x, knots_y
        return self

    def predict(self, probability):
        if not self.knots_x:
            return _clip(probability)
        probability = float(probability)
        if probability <= self.knots_x[0]:
            return _clip(self.knots_y[0])
        if probability >= self.knots_x[-1]:
            return _clip(self.knots_y[-1])
        for index in range(1, len(self.knots_x)):
            if probability <= self.knots_x[index]:
                left_x, right_x = self.knots_x[index - 1], self.knots_x[index]
                left_y, right_y = self.knots_y[index - 1], self.knots_y[index]
                if right_x - left_x < 1e-15:
                    return _clip(right_y)
                weight = (probability - left_x) / (right_x - left_x)
                return _clip(left_y + weight * (right_y - left_y))
        return _clip(self.knots_y[-1])

    def to_dict(self):
        return {"name": self.name, "knots_x": self.knots_x, "knots_y": self.knots_y}

    @property
    def description(self):
        return "%d knots" % len(self.knots_x)


CALIBRATORS = {
    "none": IdentityCalibrator,
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
}


def make_calibrator(name):
    key = str(name).strip().lower()
    if key not in CALIBRATORS:
        raise ValueError(
            "unknown calibration method %r (choose from %s)"
            % (name, ", ".join(CALIBRATORS))
        )
    return CALIBRATORS[key]()


def calibrator_from_dict(payload):
    name = (payload or {}).get("name", "none")
    if name == "platt":
        return PlattCalibrator(
            slope=payload.get("slope", 1.0), intercept=payload.get("intercept", 0.0)
        )
    if name == "isotonic":
        return IsotonicCalibrator(
            knots_x=payload.get("knots_x") or [], knots_y=payload.get("knots_y") or []
        )
    return IdentityCalibrator()


# ----------------------------------------------------------------------
# out-of-sample fitting
# ----------------------------------------------------------------------
def grouped_folds(groups, folds=5, seed=0):
    """Split indices into folds that never break a group apart.

    Matchups from the same week share players, so a calibrator trained on
    half of a week and tested on the other half is being fed information it
    would not have had. Folding by week removes that leak.
    """
    unique = sorted(set(groups))
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)

    assignment = {group: index % folds for index, group in enumerate(shuffled)}
    buckets = [[] for _ in range(folds)]
    for index, group in enumerate(groups):
        buckets[assignment[group]].append(index)
    return [bucket for bucket in buckets if bucket]


@dataclass
class CalibrationReport:
    method: str
    raw: list = field(default_factory=list)
    calibrated: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)
    fitted: object = None
    folds: int = 0
    description: str = ""

    def scores(self, bins=5):
        return {
            "raw_brier": brier_score(self.raw, self.outcomes),
            "calibrated_brier": brier_score(self.calibrated, self.outcomes),
            "raw_log_loss": log_loss(self.raw, self.outcomes),
            "calibrated_log_loss": log_loss(self.calibrated, self.outcomes),
            "raw": murphy_decomposition(self.raw, self.outcomes, bins),
            "calibrated": murphy_decomposition(self.calibrated, self.outcomes, bins),
        }

    @property
    def improvement(self):
        """Brier points recovered. Negative means the map made things worse."""
        return brier_score(self.raw, self.outcomes) - brier_score(
            self.calibrated, self.outcomes
        )


def cross_validated_calibration(
    probabilities, outcomes, groups=None, method="platt", folds=5, seed=0
):
    """Honest out-of-sample calibrated probabilities, plus a deployable map.

    Every returned probability was produced by a calibrator that never saw
    its own matchup. The `fitted` calibrator on the report is separately
    refitted on everything, and is the one to apply to future weeks.
    """
    total = len(probabilities)
    if total == 0:
        return CalibrationReport(method=method)

    if groups is None:
        groups = list(range(total))
    folds = max(2, min(folds, len(set(groups))))

    calibrated = [None] * total
    for held_out in grouped_folds(groups, folds=folds, seed=seed):
        held = set(held_out)
        train_x = [probabilities[i] for i in range(total) if i not in held]
        train_y = [outcomes[i] for i in range(total) if i not in held]
        if len(set(train_y)) < 2:
            # A fold whose training half has only one outcome carries no
            # information; fall back to the raw probability rather than
            # inventing a fit.
            for index in held_out:
                calibrated[index] = _clip(probabilities[index])
            continue
        model = make_calibrator(method).fit(train_x, train_y)
        for index in held_out:
            calibrated[index] = model.predict(probabilities[index])

    deployable = make_calibrator(method)
    if len(set(outcomes)) >= 2:
        deployable.fit(probabilities, outcomes)

    return CalibrationReport(
        method=method,
        raw=list(probabilities),
        calibrated=calibrated,
        outcomes=list(outcomes),
        fitted=deployable,
        folds=folds,
        description=deployable.description,
    )


# ----------------------------------------------------------------------
# distributional scoring: grading the whole forecast, not just the winner
# ----------------------------------------------------------------------
def crps_from_samples(sorted_samples, actual):
    """Continuous Ranked Probability Score for a sampled distribution.

    Brier throws away almost everything the simulator produced: a 140-130
    forecast and a 140-60 forecast both collapse to "home favoured". CRPS
    grades the *whole* predictive distribution against the real-valued
    outcome, so each matchup contributes far more than one bit. That is why
    it can separate two models on 60 matchups when a hit rate cannot.

    Lower is better, and it is in the units of the outcome -- points.

    Uses the sorted-sample identity for the E|X - X'| term, which makes this
    O(n log n) rather than the O(n^2) of the naive double sum.
    """
    count = len(sorted_samples)
    if count == 0:
        return 0.0
    absolute = sum(abs(value - actual) for value in sorted_samples) / count
    weighted = sum(
        (2 * (index + 1) - count - 1) * value
        for index, value in enumerate(sorted_samples)
    )
    return absolute - weighted / (count * count)


def pit_value(sorted_samples, actual):
    """Where the outcome landed inside the forecast, on a 0-1 scale.

    The probability integral transform. If the forecast distribution is
    right, these values are uniform across many matchups. Their *spread* is
    the diagnostic that matters here: a forecast that is too narrow pushes
    outcomes into both tails, so the PIT values pile up near 0 and 1 and
    their variance rises above the uniform's 1/12.
    """
    count = len(sorted_samples)
    if count == 0:
        return 0.5
    below = 0
    for value in sorted_samples:
        if value < actual:
            below += 1
        else:
            break
    return below / count


UNIFORM_VARIANCE = 1.0 / 12.0


@dataclass
class PitSummary:
    """How well the shape of the forecast matched reality."""

    count: int
    mean: float
    variance: float
    coverage_50: float
    coverage_80: float
    coverage_95: float

    @property
    def dispersion_ratio(self):
        """>1 means forecasts were too narrow, <1 too wide.

        Derived from the PIT spread against the uniform it should have had.
        """
        if self.variance <= 0:
            return 0.0
        return math.sqrt(self.variance / UNIFORM_VARIANCE)

    @property
    def verdict(self):
        ratio = self.dispersion_ratio
        if ratio > 1.05:
            return "forecasts too narrow -- overconfident"
        if ratio < 0.95:
            return "forecasts too wide -- underconfident"
        return "spread is about right"


def pit_summary(values):
    count = len(values)
    if not count:
        return PitSummary(0, 0.5, UNIFORM_VARIANCE, 0.5, 0.8, 0.95)
    mean = sum(values) / count
    variance = sum((v - mean) ** 2 for v in values) / count

    def coverage(width):
        low, high = 0.5 - width / 2, 0.5 + width / 2
        return sum(1 for v in values if low <= v <= high) / count

    return PitSummary(
        count=count,
        mean=mean,
        variance=variance,
        coverage_50=coverage(0.50),
        coverage_80=coverage(0.80),
        coverage_95=coverage(0.95),
    )


# ----------------------------------------------------------------------
# score calibration: fixing the simulator instead of patching its output
# ----------------------------------------------------------------------
@dataclass
class ScoreCalibration:
    """Two corrections applied to every player before the week is simulated.

    Recalibrating win probabilities after the fact treats the symptom. If the
    model is overconfident it is usually because the simulated spread of team
    scores is narrower than reality, and that is measurable directly:

      mean_scale  actual mean team score over projected mean. Corrects a
                  systematic tendency to under- or over-project.
      dispersion  actual spread of (result - projection) over the spread the
                  simulator produced. Above 1 means real weeks are more
                  volatile than the model thinks.

    Scaling every player's mean by `mean_scale` scales the team total by
    exactly the same factor, and scaling every player's standard deviation
    scales the team's by the same factor too -- the correlation structure is
    untouched. So both corrections are exact at the team level.

    The decisive advantage over Platt is sample size. A season gives one
    binary outcome per matchup but *two* team scores, and a team score is a
    real number rather than one bit. These two parameters are fitted on
    roughly four times the information, from the quantity that is actually
    wrong.
    """

    mean_scale: float = 1.0
    dispersion: float = 1.0

    def apply(self, mean, sd):
        return mean * self.mean_scale, sd * self.mean_scale * self.dispersion

    def rescale_margin(self, samples, centre=None):
        """Re-centre and re-spread simulated margins without re-simulating.

        A margin is a difference of two team totals, so scaling both totals
        scales the margin's mean by `mean_scale` and its deviations by
        `mean_scale * dispersion`. That makes cross-validating these two
        parameters exact and nearly free.
        """
        if not samples:
            return []
        if centre is None:
            centre = sum(samples) / len(samples)
        shifted = centre * self.mean_scale
        spread = self.mean_scale * self.dispersion
        return [shifted + spread * (value - centre) for value in samples]

    @property
    def is_identity(self):
        return abs(self.mean_scale - 1) < 1e-9 and abs(self.dispersion - 1) < 1e-9

    @property
    def description(self):
        return "mean x%.4f, spread x%.4f" % (self.mean_scale, self.dispersion)

    def to_dict(self):
        return {"mean_scale": self.mean_scale, "dispersion": self.dispersion}


def score_calibration_from_dict(payload):
    payload = payload or {}
    return ScoreCalibration(
        mean_scale=float(payload.get("mean_scale", 1.0)),
        dispersion=float(payload.get("dispersion", 1.0)),
    )


def fit_score_calibration(rows):
    """Fit (mean_scale, dispersion) from (projected, simulated_sd, actual).

    Order matters: the mean correction is applied first, because the residual
    spread has to be measured against corrected projections or the bias leaks
    into the dispersion estimate.
    """
    rows = [row for row in rows if row[0] > 0 and row[1] > 0]
    if len(rows) < 8:
        return ScoreCalibration()

    projected = sum(row[0] for row in rows)
    actual = sum(row[2] for row in rows)
    mean_scale = actual / projected if projected > 0 else 1.0

    residuals = [row[2] - row[0] * mean_scale for row in rows]
    residual_mean = sum(residuals) / len(residuals)
    residual_sd = math.sqrt(
        sum((r - residual_mean) ** 2 for r in residuals) / (len(residuals) - 1)
    )
    simulated_sd = sum(row[1] * mean_scale for row in rows) / len(rows)

    dispersion = residual_sd / simulated_sd if simulated_sd > 0 else 1.0
    # A correction bigger than this is not a calibration, it is a different
    # model, and almost certainly a sign the sample is too small.
    dispersion = min(max(dispersion, 0.5), 2.0)
    mean_scale = min(max(mean_scale, 0.5), 2.0)
    return ScoreCalibration(mean_scale=mean_scale, dispersion=dispersion)


def cross_validated_score_calibration(rows, groups, folds=5, seed=0):
    """Fit the two corrections out-of-sample, grouped by week.

    `rows` are (projected, simulated_sd, actual, margin_samples, home_won)
    tuples. Returns (fitted_on_everything, out_of_sample_probabilities).
    """
    total = len(rows)
    if total == 0:
        return ScoreCalibration(), []

    folds = max(2, min(folds, len(set(groups))))
    probabilities = [None] * total

    for held_out in grouped_folds(groups, folds=folds, seed=seed):
        held = set(held_out)
        training = [
            (rows[i][0], rows[i][1], rows[i][2]) for i in range(total) if i not in held
        ]
        model = fit_score_calibration(training)
        for index in held_out:
            samples = rows[index][3]
            adjusted = model.rescale_margin(samples)
            wins = sum(1 for value in adjusted if value > 0)
            probabilities[index] = wins / len(adjusted) if adjusted else 0.5

    deployable = fit_score_calibration([(r[0], r[1], r[2]) for r in rows])
    return deployable, probabilities
