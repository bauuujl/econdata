"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .analyzer import analyze_league
from .calibration import calibrator_from_dict
from .devig import METHODS as DEVIG_METHODS
from .report import render_html, render_text
from .simulate import CorrelationModel
from .espn import EspnClient, EspnError, parse_league_url
from .sleeper import SleeperClient, SleeperError
from .sources import build_ensemble


def load_odds(path):
    """Read a {team: price} map. Accepts American or decimal odds."""
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("odds file must be a JSON object of team -> price")
    return data


def parse_inline_odds(values):
    """--odd 'Team Name=-150' repeated, for when a file is overkill."""
    if not values:
        return {}
    prices = {}
    for item in values:
        if "=" not in item:
            raise ValueError("expected TEAM=PRICE, got %r" % item)
        team, price = item.split("=", 1)
        prices[team.strip()] = price.strip()
    return prices


def command_leagues(args, client):
    if getattr(args, "platform", "sleeper") == "espn":
        print(
            "ESPN has no public username lookup. Open your league in a browser "
            "and copy the leagueId from the URL, then pass the whole URL:\n"
            "  ffbets --platform espn --season 2026 analyze --league "
            "'https://fantasy.espn.com/football/team?leagueId=...&seasonId=2026'",
            file=sys.stderr,
        )
        return 1
    user = client.user(args.user)
    if not user or not user.get("user_id"):
        print("No Sleeper user named %r." % args.user)
        return 1

    season = args.season or client.nfl_state().get("season")
    leagues = client.user_leagues(user["user_id"], season)
    if not leagues:
        print("%s has no %s leagues (try --season)." % (user.get("display_name"), season))
        return 1

    print("Leagues for %s in %s:" % (user.get("display_name"), season))
    for league in leagues:
        print(
            "  %-22s %s  (%s teams, %s)"
            % (
                league["league_id"],
                league.get("name", "?"),
                league.get("total_rosters"),
                league.get("status"),
            )
        )
    print("\nAnalyze one with:  ffbets analyze --league <id>")
    return 0


def load_json(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_calibrator(path):
    """A calibration map saved by `backtest --save-calibrator`."""
    if not path:
        return None
    return calibrator_from_dict(load_json(path))


def load_correlation(args):
    """Fitted loadings from `correlations --save`, or the CLI defaults."""
    saved = getattr(args, "correlations", None)
    if saved:
        payload = load_json(saved)
        return CorrelationModel(
            offense=payload.get("loadings", {}).get("WR", 0.35),
            kicker=payload.get("loadings", {}).get("K", 0.20),
            defense=payload.get("defense", 0.30),
            game_share=payload.get("game_share", 0.35),
            loadings=payload.get("loadings"),
        )
    return CorrelationModel(
        offense=args.correlation, defense=args.correlation * 0.85
    )


def make_ensemble(args):
    return build_ensemble(
        args.sources,
        file_path=getattr(args, "projection_file", None),
        disagreement_weight=getattr(args, "disagreement", 1.0),
    )


def command_analyze(args, client):
    book_odds = load_odds(args.odds) or {}
    book_odds.update(parse_inline_odds(args.odd))

    analysis = analyze_league(
        client,
        args.league,
        week=args.week,
        n_sims=args.sims,
        seed=args.seed,
        lineup_mode=args.lineup,
        projection_weight=args.projection_weight,
        history_weeks=args.history_weeks,
        correlation=load_correlation(args),
        use_player_db=not args.no_player_db,
        book_odds=book_odds or None,
        kelly_multiplier=args.kelly,
        ensemble=make_ensemble(args),
        calibrator=load_calibrator(args.calibrator),
        devig_method=args.devig,
        shrinkage=args.shrinkage,
        exposure_cap=args.cap,
        verbose=args.verbose,
    )
    return emit(analysis, args)


def command_demo(args, client):
    from .demo import build_demo_analysis, demo_book_odds

    analysis = build_demo_analysis(
        n_sims=args.sims, seed=args.seed or 7, kelly_multiplier=args.kelly
    )

    book_odds = load_odds(args.odds) or {}
    book_odds.update(parse_inline_odds(args.odd))
    if not book_odds and not args.no_book:
        book_odds = demo_book_odds(analysis)
        analysis.notes.append("book prices are simulated too (use --odds for real ones)")

    if book_odds:
        from .analyzer import build_recommendations

        analysis.recommendations = build_recommendations(
            analysis.matchups, book_odds, analysis.n_sims, args.kelly
        )
    return emit(analysis, args)


def command_backtest(args, client):
    from .backtest import backtest_league, render_backtest, render_backtest_html
    from .clv import load_lines

    methods = {
        "both": ("platt", "isotonic"),
        "platt": ("platt",),
        "isotonic": ("isotonic",),
        "none": (),
    }[args.calibration]

    result = backtest_league(
        client,
        args.league,
        first_week=args.first_week,
        last_week=args.last_week,
        n_sims=args.sims,
        seed=args.seed or 1,
        history_weeks=args.history_weeks,
        projection_weight=args.projection_weight,
        correlation=load_correlation(args),
        ensemble=make_ensemble(args),
        calibration_methods=methods,
        calibration_folds=args.folds,
        devig_method=args.devig,
        lines=load_lines(load_json(args.lines)) if args.lines else None,
        kelly=args.kelly,
        bins=args.bins,
        market_mode=args.market,
        verbose=args.verbose,
    )
    print(render_backtest(result))

    if args.save_calibrator:
        report = result.calibrations.get(result.chosen_method)
        payload = report.fitted.to_dict() if report and report.fitted else {"name": "none"}
        with open(args.save_calibrator, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(
            "\nCalibration map (%s) written to %s"
            % (result.chosen_method, os.path.abspath(args.save_calibrator))
        )

    if args.html:
        with open(args.html, "w", encoding="utf-8") as handle:
            handle.write(render_backtest_html(result))
        print("\nHTML report written to %s" % os.path.abspath(args.html))
    return 0


def command_correlations(args, client):
    from .correlations import estimate_correlations

    league = client.league(args.league)
    try:
        player_db = client.players()
    except Exception:
        player_db = {}

    fit = estimate_correlations(
        client,
        league["scoring_settings"],
        league["season"],
        range(args.first_week, args.last_week + 1),
        player_db=player_db,
        verbose=args.verbose,
    )

    print("=" * 78)
    print("  CORRELATIONS FITTED FROM GAME LOGS  |  %s season" % league["season"])
    print("=" * 78)
    print("  weeks %d-%d" % (min(fit.weeks), max(fit.weeks)))
    print("")
    print("OBSERVED  (correlation of fantasy points, then of the normals beneath)")
    print("-" * 78)
    print("  %-9s %-11s %9s %9s %7s" % ("KIND", "PAIR", "POINTS", "LATENT", "N"))
    for item in sorted(fit.observations, key=lambda o: (o.kind, -abs(o.observed))):
        print(
            "  %-9s %-11s %+9.3f %+9.3f %7d"
            % (item.kind, "%s-%s" % (item.role_a, item.role_b),
               item.observed, item.latent, item.count)
        )

    print("")
    print("FITTED LOADINGS")
    print("-" * 78)
    for position, value in sorted(fit.loadings.items()):
        print("  rho_%-6s %.3f" % (position, value))
    print("  game_share  %.3f   (share of an offence's day owned by the game)" % fit.game_share)
    print("  defense     %.3f   (how hard a defence loads against the offence it faces)" % fit.defense)
    print("  rms misfit  %.4f" % fit.residual)

    print("")
    print("WORST MISFITS  (where a one-factor-per-offence model cannot reach)")
    print("-" * 78)
    print("  %-9s %-11s %9s %9s %8s" % ("KIND", "PAIR", "OBSERVED", "MODEL", "MISS"))
    for row in fit.residuals()[:5]:
        print(
            "  %-9s %-11s %+9.3f %+9.3f %+8.3f"
            % (row["kind"], row["pair"], row["observed"], row["implied"], row["residual"])
        )
    print("")
    print("  A single shared factor per offence can only produce POSITIVE teammate")
    print("  correlation, so genuinely negative pairs -- receivers competing for the")
    print("  same targets -- are unrepresentable and surface here rather than being")
    print("  absorbed silently into the loadings.")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(fit.to_dict(), handle, indent=2)
        print("\nSaved to %s (pass it back with --correlations)" % os.path.abspath(args.save))
    return 0


def command_stress(args, client):
    from .staking import render_stress, stress_test

    levels = [
        (level, stress_test(overconfidence=level, seasons=args.seasons,
                            weeks=args.weeks, bets_per_week=args.bets,
                            vig=args.vig, seed=args.seed or 0))
        for level in args.overconfidence
    ]
    print(render_stress(levels))
    return 0


def emit(analysis, args):
    if args.json:
        print(json.dumps(to_dict(analysis), indent=2))
    else:
        print(render_text(analysis, detail=args.detail, top=args.top))

    if args.html:
        with open(args.html, "w", encoding="utf-8") as handle:
            handle.write(render_html(analysis))
        print("\nHTML report written to %s" % os.path.abspath(args.html))
    return 0


def to_dict(analysis):
    """A machine-readable dump of the whole analysis."""
    return {
        "league": analysis.league_name,
        "season": analysis.season,
        "week": analysis.week,
        "scoring": analysis.scoring,
        "simulations": analysis.n_sims,
        "lineup_mode": analysis.lineup_mode,
        "notes": analysis.notes,
        "matchups": [
            {
                "matchup_id": matchup.matchup_id,
                "teams": [
                    {
                        "roster_id": team.roster_id,
                        "name": team.name,
                        "owner": team.owner,
                        "projected": round(team.mean, 2),
                        "sd": round(team.sd, 2),
                        "p10": round(team.p10, 2),
                        "p50": round(team.p50, 2),
                        "p90": round(team.p90, 2),
                        "win_probability": round(probability, 4),
                        "starters": [
                            {
                                "name": player.name,
                                "position": player.position,
                                "team": player.nfl_team,
                                "opponent": player.opponent,
                                "mean": round(player.mean, 2),
                                "sd": round(player.sd, 2),
                                "play_probability": player.play_prob,
                                "source": player.source,
                            }
                            for player in team.starters
                        ],
                    }
                    for team, probability in (
                        (matchup.home, matchup.p_home),
                        (matchup.away, matchup.p_away),
                    )
                ],
                "tie_probability": round(matchup.p_tie, 4),
                "median_margin": round(matchup.median_margin, 2),
                "margin_p10": round(matchup.margin_p10, 2),
                "margin_p90": round(matchup.margin_p90, 2),
            }
            for matchup in analysis.matchups
        ],
        "recommendations": [
            {
                "team": item.team,
                "opponent": item.opponent,
                "model_probability": round(item.model_probability, 4),
                "fair_american": item.fair_american,
                "book_american": item.price_american,
                "market_probability": (
                    round(item.devig_probability, 4)
                    if item.devig_probability is not None
                    else (
                        round(item.implied_probability, 4)
                        if item.implied_probability is not None
                        else None
                    )
                ),
                "edge": round(item.edge, 4) if item.edge is not None else None,
                "ev_per_dollar": (
                    round(item.expected_value, 4)
                    if item.expected_value is not None
                    else None
                ),
                "kelly_stake": round(item.kelly, 4) if item.kelly is not None else None,
            }
            for item in analysis.recommendations
        ],
    }


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ffbets",
        description="Simulate a fantasy football week and find the bettable edge.",
    )
    parser.add_argument("--version", action="version", version="ffbets " + __version__)
    parser.add_argument("--cache-dir", help="where to cache API responses")
    parser.add_argument(
        "--platform",
        choices=["sleeper", "espn"],
        default="sleeper",
        help="which fantasy site the league lives on (default: sleeper)",
    )
    parser.add_argument(
        "--season", help="season year, required for ESPN (e.g. 2026)"
    )
    parser.add_argument(
        "--espn-s2",
        help="ESPN espn_s2 cookie; ESPN needs it for every league, even public "
        "ones. Defaults to the ESPN_S2 environment variable.",
    )
    parser.add_argument(
        "--swid", help="ESPN SWID cookie. Defaults to the ESPN_SWID variable."
    )
    parser.add_argument(
        "--offline", action="store_true", help="use only cached data; never hit network"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--devig",
        choices=list(DEVIG_METHODS),
        default="shin",
        help="how to strip the book's margin (default: shin)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    def add_output_flags(target):
        target.add_argument("--sims", type=int, default=20000, help="simulations (default 20000)")
        target.add_argument("--seed", type=int, help="seed for reproducible runs")
        target.add_argument("--odds", help="JSON file of team -> sportsbook price")
        target.add_argument(
            "--odd",
            action="append",
            metavar="TEAM=PRICE",
            help="inline price, repeatable (e.g. --odd 'Sunday Scaries=-135')",
        )
        target.add_argument("--kelly", type=float, default=0.25, help="Kelly multiplier")
        target.add_argument("--top", type=int, default=8, help="rows in the bets table")
        target.add_argument("--detail", action="store_true", help="show starting lineups")
        target.add_argument("--html", metavar="PATH", help="also write an HTML report")
        target.add_argument("--json", action="store_true", help="emit JSON instead of text")

    def add_model_flags(target):
        target.add_argument(
            "--sources",
            default="rotowire,form",
            help="projection sources, comma separated with optional weights "
            "(e.g. 'rotowire:0.7,form:0.2,opponent:0.1')",
        )
        target.add_argument(
            "--projection-file", help="your own projections, for source 'file'"
        )
        target.add_argument(
            "--disagreement",
            type=float,
            default=1.0,
            help="how hard source disagreement widens a player's spread",
        )
        target.add_argument(
            "--correlations",
            metavar="PATH",
            help="fitted loadings from the `correlations` command",
        )
        target.add_argument(
            "--correlation",
            type=float,
            default=0.35,
            help="flat teammate correlation when --correlations is not given",
        )
        target.add_argument(
            "--projection-weight",
            type=float,
            default=0.80,
            help="weight on projections vs a player's own game log (default 0.80)",
        )
        target.add_argument(
            "--history-weeks", type=int, default=8, help="weeks of game log to read"
        )

    leagues = sub.add_parser("leagues", help="find your league IDs from a username")
    leagues.add_argument("--user", required=True, help="your Sleeper username")
    leagues.add_argument("--season", help="season to list (default: current)")
    leagues.set_defaults(handler=command_leagues)

    analyze = sub.add_parser("analyze", help="analyze a league week")
    analyze.add_argument("--league", required=True, help="Sleeper league ID")
    analyze.add_argument("--week", type=int, help="week to analyze (default: current)")
    analyze.add_argument(
        "--lineup",
        choices=["auto", "declared", "optimal"],
        default="auto",
        help="whose lineup to trust (default: auto)",
    )
    analyze.add_argument(
        "--no-player-db",
        action="store_true",
        help="skip the 14 MB player database (faster, sparser names)",
    )
    analyze.add_argument(
        "--calibrator",
        metavar="PATH",
        help="a calibration map from `backtest --save-calibrator`; applied "
        "before EV and Kelly",
    )
    analyze.add_argument(
        "--shrinkage",
        type=float,
        default=0.0,
        help="weight given to the market's de-vigged probability when sizing "
        "(0.3 is a sensible guard against miscalibration)",
    )
    analyze.add_argument(
        "--cap", type=float, default=0.25, help="ceiling on a single stake"
    )
    add_model_flags(analyze)
    add_output_flags(analyze)
    analyze.set_defaults(handler=command_analyze)

    backtest = sub.add_parser(
        "backtest", help="grade the model against weeks already played"
    )
    backtest.add_argument("--league", required=True, help="Sleeper league ID")
    backtest.add_argument("--first-week", type=int, default=2)
    backtest.add_argument(
        "--last-week", type=int, help="default: the week before the playoffs"
    )
    backtest.add_argument("--sims", type=int, default=4000)
    backtest.add_argument("--seed", type=int)
    backtest.add_argument(
        "--calibration",
        choices=["both", "platt", "isotonic", "none"],
        default="both",
        help="which recalibration maps to fit (default: both)",
    )
    backtest.add_argument(
        "--folds", type=int, default=5, help="cross-validation folds, grouped by week"
    )
    backtest.add_argument(
        "--bins", type=int, default=5, help="buckets in the reliability curve"
    )
    backtest.add_argument(
        "--lines",
        metavar="PATH",
        help="real opening/closing prices: {week: {team: {open, close}}}",
    )
    backtest.add_argument(
        "--market",
        choices=["sharp", "model"],
        default="sharp",
        help="how to simulate a market when --lines is absent (default: sharp)",
    )
    backtest.add_argument("--kelly", type=float, default=0.25)
    backtest.add_argument(
        "--save-calibrator", metavar="PATH", help="write the fitted map for `analyze`"
    )
    backtest.add_argument("--html", metavar="PATH", help="write an HTML report")
    add_model_flags(backtest)
    backtest.set_defaults(handler=command_backtest)

    correlations = sub.add_parser(
        "correlations", help="fit the simulator's correlations from game logs"
    )
    correlations.add_argument("--league", required=True, help="Sleeper league ID")
    correlations.add_argument("--first-week", type=int, default=1)
    correlations.add_argument("--last-week", type=int, default=13)
    correlations.add_argument("--save", metavar="PATH", help="write the fitted loadings")
    correlations.set_defaults(handler=command_correlations)

    stress = sub.add_parser(
        "stress", help="what miscalibration costs each staking rule"
    )
    stress.add_argument(
        "--overconfidence",
        type=float,
        nargs="+",
        default=[1.0, 1.4, 1.8],
        help="log-odds stretch factors to test (1.0 = perfectly calibrated)",
    )
    stress.add_argument("--seasons", type=int, default=400)
    stress.add_argument("--weeks", type=int, default=14)
    stress.add_argument("--bets", type=int, default=5, help="bets per week")
    stress.add_argument("--vig", type=float, default=0.045)
    stress.add_argument("--seed", type=int)
    stress.set_defaults(handler=command_stress)

    demo = sub.add_parser("demo", help="run on synthetic data, no network needed")
    demo.add_argument(
        "--no-book", action="store_true", help="skip the simulated sportsbook prices"
    )
    add_output_flags(demo)
    demo.set_defaults(handler=command_demo)

    return parser


def make_client(args):
    """Build the platform client, normalising an ESPN league URL if given."""
    if args.platform == "espn":
        league_id = getattr(args, "league", None)
        season = args.season
        if league_id:
            league_id, url_season = parse_league_url(league_id)
            args.league = league_id
            season = season or url_season
        if not season:
            raise ValueError(
                "ESPN needs --season (its API is keyed by season year, and the "
                "league endpoint has no way to tell you the current one)"
            )
        return EspnClient(
            league_id=league_id,
            season=season,
            espn_s2=args.espn_s2,
            swid=args.swid,
            cache_dir=args.cache_dir,
            offline=args.offline,
            verbose=args.verbose,
        )
    return SleeperClient(
        cache_dir=args.cache_dir, offline=args.offline, verbose=args.verbose
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        client = make_client(args)
    except ValueError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1
    try:
        return args.handler(args, client)
    except (SleeperError, EspnError) as exc:
        print("%s" % exc, file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
