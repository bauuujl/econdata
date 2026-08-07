"""A small, cached client for Sleeper's public read-only API.

Sleeper's read endpoints need no key and no auth, which is what makes this
tool work out of the box.  They are also rate limited, so every response is
cached on disk; a repeated run costs no network at all.

Docs: https://docs.sleeper.com/
"""

from __future__ import annotations

import gzip
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.sleeper.app"
USER_AGENT = "ffbets/1.0 (+https://github.com/bauuujl/econdata)"

# Cache lifetimes, in seconds.  Anything from a completed week is immutable,
# but lineups and projections churn right up to kickoff.
TTL_STATIC = 7 * 24 * 3600  # player database, finished-week stats
TTL_LEAGUE = 6 * 3600  # league settings, rosters, users
TTL_LIVE = 15 * 60  # matchups, projections, NFL state


class SleeperError(RuntimeError):
    """Raised when the API is unreachable or answers with an error status."""


class SleeperClient:
    def __init__(self, cache_dir=None, offline=False, timeout=30, verbose=False):
        if cache_dir is None:
            cache_dir = os.path.join(
                os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
                "ffbets",
            )
        self.cache_dir = cache_dir
        self.offline = offline
        self.timeout = timeout
        self.verbose = verbose
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------
    def _cache_path(self, key):
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return os.path.join(self.cache_dir, safe[:180] + ".json.gz")

    def _read_cache(self, key, ttl):
        path = self._cache_path(key)
        try:
            age = time.time() - os.path.getmtime(path)
        except OSError:
            return None
        # In offline mode a stale cache still beats no data at all.
        if age > ttl and not self.offline:
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return None

    def _write_cache(self, key, payload):
        path = self._cache_path(key)
        try:
            # Write-then-rename so an interrupted run can't leave a truncated
            # cache entry that later reads would trip over.
            temp = path + ".tmp"
            with gzip.open(temp, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temp, path)
        except OSError:
            pass

    def get(self, path, params=None, ttl=TTL_LIVE, cache_key=None):
        url = BASE + path
        if params:
            url += "?" + urllib.parse.urlencode(params, doseq=True)
        key = cache_key or url[len(BASE) :]

        cached = self._read_cache(key, ttl)
        if cached is not None:
            return cached
        if self.offline:
            raise SleeperError(
                "offline mode: no cached response for %s (run once with network "
                "access to populate the cache)" % path
            )

        if self.verbose:
            print("  fetching %s" % url)
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            raise SleeperError("%s returned HTTP %s" % (url, exc.code)) from exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise SleeperError("could not reach %s: %s" % (url, exc)) from exc

        self._write_cache(key, payload)
        return payload

    # ------------------------------------------------------------------
    # endpoints
    # ------------------------------------------------------------------
    def nfl_state(self):
        """Current season, week and season type."""
        return self.get("/v1/state/nfl", ttl=TTL_LIVE)

    def user(self, username_or_id):
        return self.get("/v1/user/%s" % username_or_id, ttl=TTL_LEAGUE)

    def user_leagues(self, user_id, season):
        return self.get(
            "/v1/user/%s/leagues/nfl/%s" % (user_id, season), ttl=TTL_LEAGUE
        )

    def league(self, league_id):
        return self.get("/v1/league/%s" % league_id, ttl=TTL_LEAGUE)

    def league_users(self, league_id):
        return self.get("/v1/league/%s/users" % league_id, ttl=TTL_LEAGUE)

    def league_rosters(self, league_id):
        return self.get("/v1/league/%s/rosters" % league_id, ttl=TTL_LEAGUE)

    def matchups(self, league_id, week):
        return self.get("/v1/league/%s/matchups/%s" % (league_id, week), ttl=TTL_LIVE)

    def players(self):
        """The full NFL player database (~14 MB), cached for a week."""
        return self.get("/v1/players/nfl", ttl=TTL_STATIC, cache_key="players_nfl")

    def week_stats(self, season, week, season_type="regular"):
        """Actual production for a completed week: {player_id: statline}."""
        return self.get(
            "/v1/stats/nfl/%s/%s/%s" % (season_type, season, week), ttl=TTL_STATIC
        )

    def week_projections(self, season, week, season_type="regular", positions=None):
        """Rotowire projections for a week, as a list of per-player rows.

        This is the endpoint Sleeper's own app uses.  It is public but not
        formally documented, so callers should tolerate it returning nothing.
        """
        if positions is None:
            positions = ["QB", "RB", "WR", "TE", "K", "DEF"]
        params = [("season_type", season_type), ("order_by", "ppr")]
        params += [("position[]", position) for position in positions]
        return self.get(
            "/projections/nfl/%s/%s" % (season, week),
            params=params,
            ttl=TTL_LIVE,
            cache_key="proj_%s_%s_%s" % (season, week, season_type),
        )
