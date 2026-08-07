"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .analyzer import analyze_league
from .report import render_html, render_text
from .simulate import CorrelationModel
from .sleeper import SleeperClient, SleeperError


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
        correlation=CorrelationModel(
            offense=args.correlation, defense=args.correlation * 0.85
        ),
        use_player_db=not args.no_player_db,
        book_odds=book_odds or None,
        kelly_multiplier=args.kelly,
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
    from .backtest import backtest_league, render_backtest

    result = backtest_league(
        client,
        args.league,
        first_week=args.first_week,
        last_week=args.last_week,
        n_sims=args.sims,
        seed=args.seed or 1,
        history_weeks=args.history_weeks,
        projection_weight=args.projection_weight,
        correlation=CorrelationModel(
            offense=args.correlation, defense=args.correlation * 0.85
        ),
        verbose=args.verbose,
    )
    print(render_backtest(result))
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
        "--offline", action="store_true", help="use only cached data; never hit network"
    )
    parser.add_argument("-v", "--verbose", action="store_true")

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
        "--projection-weight",
        type=float,
        default=0.80,
        help="weight on projections vs a player's own game log (default 0.80)",
    )
    analyze.add_argument(
        "--history-weeks", type=int, default=8, help="weeks of game log to read"
    )
    analyze.add_argument(
        "--correlation",
        type=float,
        default=0.35,
        help="how tightly teammates move together, 0-0.9 (default 0.35)",
    )
    analyze.add_argument(
        "--no-player-db",
        action="store_true",
        help="skip the 14 MB player database (faster, sparser names)",
    )
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
    backtest.add_argument("--history-weeks", type=int, default=8)
    backtest.add_argument("--projection-weight", type=float, default=0.80)
    backtest.add_argument("--correlation", type=float, default=0.35)
    backtest.set_defaults(handler=command_backtest)

    demo = sub.add_parser("demo", help="run on synthetic data, no network needed")
    demo.add_argument(
        "--no-book", action="store_true", help="skip the simulated sportsbook prices"
    )
    add_output_flags(demo)
    demo.set_defaults(handler=command_demo)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    client = SleeperClient(
        cache_dir=args.cache_dir, offline=args.offline, verbose=args.verbose
    )
    try:
        return args.handler(args, client)
    except SleeperError as exc:
        print("Sleeper API error: %s" % exc, file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
