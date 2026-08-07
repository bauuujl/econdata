"""Projection sources, and combining several of them.

One projection feed is one opinion. Rotowire's numbers (via Sleeper) are
good, but they are a single model with a single set of blind spots, and a
model built on top of them inherits every one.

An ensemble helps twice over. The obvious benefit is accuracy: averaging
forecasts that err in different directions cancels part of the error. The
less obvious one matters more here -- **disagreement between sources is
itself information**. When two sources say 14 and one says 6, the honest
description of that player is not "11.3", it is "11.3, and we do not really
know". That extra uncertainty is folded into the player's spread, which
widens the simulated distribution and correctly pulls extreme win
probabilities back toward the middle.

Sources available without a paid feed:

  rotowire   Sleeper's published projections, re-scored under league rules.
  form       exponentially-weighted recent scoring. Independent of any
             projection provider, and a genuinely strong baseline.
  opponent   recent form adjusted for how generous the opposing defence has
             been to that position, using the points-allowed-by-position
             figures in the defensive statlines.
  file       your own numbers, from a JSON or CSV export.

Note on other providers: Sleeper's projections endpoint accepts a `company`
parameter and will echo back names like `sportradar` or `fantasypros`, but
only `rotowire` returns populated statlines -- the others come back empty.
Do not mistake a 200 response for a second opinion.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field

# Positions whose points-allowed figures the defensive statlines expose.
ALLOWED_KEYS = {
    "QB": "fan_pts_allow_qb",
    "RB": "fan_pts_allow_rb",
    "WR": "fan_pts_allow_wr",
    "TE": "fan_pts_allow_te",
    "K": "fan_pts_allow_k",
}


@dataclass
class Estimate:
    """What one source thinks, and how confident the ensemble should be."""

    mean: float
    disagreement: float = 0.0
    contributions: dict = field(default_factory=dict)
    label: str = ""


class ProjectionSource:
    """A named opinion about how many points a player will score."""

    name = "source"
    short = "src"

    def __init__(self, weight=1.0):
        self.weight = float(weight)

    def project(self, player_id, context):
        """Points, or None if this source has nothing to say."""
        raise NotImplementedError


class RotowireSource(ProjectionSource):
    """Sleeper's published projections, scored under the league's rules."""

    name = "rotowire"
    short = "proj"

    def project(self, player_id, context):
        entry = context.projections.get(player_id)
        if not entry or not entry.get("has_stats"):
            return None
        return entry["points"]


class RecentFormSource(ProjectionSource):
    """Exponentially-weighted average of a player's own recent scoring.

    A half-life of three games is roughly where fantasy scoring stops
    rewarding older data: recent enough to catch a change in role, long
    enough not to chase a single outlier.
    """

    name = "form"
    short = "form"

    def __init__(self, weight=1.0, half_life=3.0, min_games=3):
        super().__init__(weight)
        self.half_life = float(half_life)
        self.min_games = int(min_games)

    def project(self, player_id, context):
        history = context.history.get(player_id) or []
        if len(history) < self.min_games:
            return None
        decay = 0.5 ** (1.0 / self.half_life)
        total = weight_sum = 0.0
        # History runs oldest to newest, so the most recent game gets weight 1.
        for age, points in enumerate(reversed(history)):
            weight = decay ** age
            total += weight * points
            weight_sum += weight
        return total / weight_sum if weight_sum else None


class OpponentAdjustedSource(ProjectionSource):
    """Recent form, scaled by how generous the opposing defence has been.

    The scale factor is shrunk halfway toward 1. Points allowed by position
    is a noisy measure over a handful of games, and taking it at face value
    would swing projections around far more than the signal justifies.
    """

    name = "opponent"
    short = "opp"

    def __init__(self, weight=1.0, half_life=3.0, min_games=3, shrinkage=0.5):
        super().__init__(weight)
        self.form = RecentFormSource(weight, half_life, min_games)
        self.shrinkage = float(shrinkage)

    def project(self, player_id, context):
        baseline = self.form.project(player_id, context)
        if baseline is None:
            return None

        entry = context.projections.get(player_id) or {}
        opponent = entry.get("opponent")
        position = entry.get("position") or (
            context.player_db.get(player_id, {}) or {}
        ).get("position")
        if not opponent or position not in ALLOWED_KEYS:
            return None

        factor = context.defense_factor(opponent, position)
        if factor is None:
            return None
        return baseline * (1.0 + self.shrinkage * (factor - 1.0))


class FileSource(ProjectionSource):
    """Your own projections, keyed by Sleeper player ID or by player name."""

    name = "file"
    short = "file"

    def __init__(self, path, weight=1.0):
        super().__init__(weight)
        self.path = path
        self.by_id = {}
        self.by_name = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            raise ValueError("projection file not found: %s" % self.path)

        if self.path.lower().endswith(".csv"):
            with open(self.path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    self._add(row.get("player_id"), row.get("name"), row.get("points"))
        else:
            with open(self.path, encoding="utf-8") as handle:
                payload = json.load(handle)
            for key, value in (payload or {}).items():
                self._add(key, key, value)

    def _add(self, player_id, name, points):
        try:
            value = float(points)
        except (TypeError, ValueError):
            return
        if player_id:
            self.by_id[str(player_id).strip()] = value
        if name:
            self.by_name[str(name).strip().lower()] = value

    def project(self, player_id, context):
        if player_id in self.by_id:
            return self.by_id[player_id]
        entry = context.projections.get(player_id) or {}
        name = (entry.get("name") or "").strip().lower()
        return self.by_name.get(name)


class Ensemble:
    """A weighted committee of sources.

    Sources that decline to project a player simply do not vote, and the
    remaining weights are renormalised -- so a source that only covers
    starters does not drag bench projections toward zero.
    """

    def __init__(self, sources, disagreement_weight=1.0):
        self.sources = list(sources)
        self.disagreement_weight = float(disagreement_weight)

    @property
    def names(self):
        return [source.name for source in self.sources]

    def combine(self, player_id, context):
        votes = []
        for source in self.sources:
            if source.weight <= 0:
                continue
            try:
                value = source.project(player_id, context)
            except Exception:
                value = None
            if value is not None and math.isfinite(value):
                votes.append((source, float(value)))

        if not votes:
            return None

        total_weight = sum(source.weight for source, _ in votes)
        if total_weight <= 0:
            return None

        mean = sum(source.weight * value for source, value in votes) / total_weight
        variance = (
            sum(source.weight * (value - mean) ** 2 for source, value in votes)
            / total_weight
        )

        return Estimate(
            mean=mean,
            disagreement=math.sqrt(variance) * self.disagreement_weight,
            contributions={source.name: round(value, 2) for source, value in votes},
            label="+".join(source.short for source, _ in votes),
        )


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------
DEFAULT_WEIGHTS = {"rotowire": 0.80, "form": 0.20, "opponent": 0.20, "file": 0.50}

BUILDERS = {
    "rotowire": RotowireSource,
    "form": RecentFormSource,
    "opponent": OpponentAdjustedSource,
}


def build_ensemble(spec="rotowire,form", file_path=None, disagreement_weight=1.0):
    """Parse a source spec into an Ensemble.

    The spec is a comma-separated list, each entry optionally carrying a
    weight after a colon: `rotowire:0.7,form:0.2,opponent:0.1`. Weights are
    relative and get renormalised per player, so they need not sum to 1.
    """
    sources = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        name, _, weight_text = item.partition(":")
        name = name.strip().lower()
        weight = float(weight_text) if weight_text else DEFAULT_WEIGHTS.get(name, 1.0)

        if name == "file":
            if not file_path:
                raise ValueError("source 'file' needs --projection-file")
            sources.append(FileSource(file_path, weight))
        elif name in BUILDERS:
            sources.append(BUILDERS[name](weight))
        else:
            raise ValueError(
                "unknown projection source %r (choose from %s, file)"
                % (name, ", ".join(sorted(BUILDERS)))
            )

    if not sources:
        raise ValueError("no projection sources selected")
    return Ensemble(sources, disagreement_weight=disagreement_weight)
