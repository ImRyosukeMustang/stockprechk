"""
scheduler.py — runs main.run_pipeline_sma_only() on a recurring timer via APScheduler.

This is a thin wrapper, deliberately. All the actual work (fetch, indicators,
decide, allocate) lives in main.py; scheduler.py's only job is to
call it on a schedule and keep the process alive. If APScheduler isn't
installed, this module says so clearly and exits rather than pretending to
run — there's no meaningful degraded mode for "the thing whose entire job
is scheduling."

Usage:
    python scheduler.py                # run forever, every config.PIPELINE_INTERVAL_MINUTES
    python scheduler.py --once         # run the pipeline once immediately and exit (no scheduling)
    python scheduler.py --interval 15  # override the interval for this run
"""

from __future__ import annotations

import argparse
import sys

import config
import database
import journal
import main as pipeline_main

log = config.get_logger(__name__)


def run_full_cycle(tickers: list[str] | None = None) -> None:
    """
    Run one complete SMA-only cycle: fetch daily data, compute indicators,
    persist deterministic SMA signals, and calculate portfolio allocation.
    """
    tickers = tickers or config.WATCHLIST
    pipeline_main.run_pipeline_sma_only(tickers)
    with database.get_connection() as conn:
        journal.backfill_forward_returns(conn)
        journal.print_journal_summary(conn)


def run_once(tickers: list[str] | None = None) -> None:
    log.info("Running one pipeline cycle immediately (--once).")
    run_full_cycle(tickers)
    log.info("One-off cycle complete.")


def run_forever(interval_minutes: int, tickers: list[str] | None = None) -> None:
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        log.error(
            "apscheduler is not installed — cannot run on a schedule. "
            "`pip install apscheduler`, or use `python scheduler.py --once` to run a single cycle without it."
        )
        sys.exit(1)

    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_full_cycle,
        "interval",
        minutes=interval_minutes,
        kwargs={"tickers": tickers},
        next_run_time=None,  # don't fire immediately on startup; wait one interval first
        id="stockoracle_pipeline",
        max_instances=1,       # never let a slow cycle overlap with the next tick
        coalesce=True,         # if a tick was missed (process was asleep), run once, not N times
    )
    log.info(
        "StockOracle scheduler started — running every %d minute(s) for %s. Press Ctrl+C to stop. (DRY_RUN=%s)",
        interval_minutes, ", ".join(tickers or config.WATCHLIST), config.DRY_RUN,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the StockOracle pipeline on a schedule.")
    parser.add_argument("--once", action="store_true", help="Run a single pipeline cycle immediately and exit.")
    parser.add_argument("--interval", type=int, default=config.PIPELINE_INTERVAL_MINUTES, help="Minutes between runs.")
    parser.add_argument("tickers", nargs="*", help="Optional specific tickers (default: full watchlist).")
    args = parser.parse_args()

    requested_tickers = [t.upper() for t in args.tickers] or None

    if args.once:
        run_once(requested_tickers)
    else:
        run_forever(args.interval, requested_tickers)
