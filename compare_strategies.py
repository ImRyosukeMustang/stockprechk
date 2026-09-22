"""
compare_strategies.py

Compares three things, per ticker, over the same backtest window:
    1. SMA 50/200 crossover baseline (from backtests/sma_baseline_results.json)
    2. ML (XGBoost) strategy (from the backtest_results table in stockoracle.db)
    3. Buy-and-hold (embedded in both of the above as a benchmark column)

Prints a side-by-side table and a verdict on whether the ML model's added
complexity is actually earning its keep versus a 70-year-old technical rule.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import config

logger = config.get_logger(__name__)

SMA_RESULTS_PATH = Path("backtests") / "sma_baseline_results.json"
DB_PATH = Path("stockoracle.db")


def load_sma_results() -> dict:
    """Load the SMA baseline results produced by sma_baseline.py."""
    if not SMA_RESULTS_PATH.exists():
        logger.error(
            "%s not found. Run sma_baseline.py first.", SMA_RESULTS_PATH
        )
        return {}

    with open(SMA_RESULTS_PATH, "r") as f:
        return json.load(f)


def load_ml_results() -> dict:
    """Load the most recent ML backtest row per ticker from stockoracle.db."""
    if not DB_PATH.exists():
        logger.error("%s not found.", DB_PATH)
        return {}

    query = """
        SELECT
            ticker,
            strategy,
            start_date,
            end_date,
            num_trades,
            win_rate,
            strategy_return_pct,
            benchmark_return_pct,
            max_drawdown_pct,
            sharpe_ratio
        FROM backtest_results
        WHERE rowid IN (
            SELECT MAX(rowid)
            FROM backtest_results
            GROUP BY ticker
        )
    """

    results: dict[str, dict] = {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            for row in cursor.fetchall():
                ticker = row["ticker"]
                results[ticker] = {
                    "ticker": ticker,
                    "strategy": row["strategy"],
                    "start_date": row["start_date"],
                    "end_date": row["end_date"],
                    "n_trades": row["num_trades"],
                    "win_rate": row["win_rate"],
                    "total_return_pct": row["strategy_return_pct"],
                    "buy_hold_return_pct": row["benchmark_return_pct"],
                    "max_drawdown_pct": row["max_drawdown_pct"],
                    "sharpe_ratio": row["sharpe_ratio"],
                }
    except sqlite3.Error:
        logger.exception("Failed to read backtest_results from %s", DB_PATH)
        return {}

    return results


def _fmt_pct(value) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_num(value, decimals: int = 2) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "n/a"


def print_comparison(sma_results: dict, ml_results: dict) -> None:
    """Print a side-by-side comparison table for each ticker in the watchlist."""
    header = (
        f"{'Ticker':<10}{'Strategy':<16}{'Return':>9}{'B&H':>9}"
        f"{'Trades':>8}{'Win%':>8}{'Sharpe':>8}{'MaxDD':>8}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("STRATEGY COMPARISON: SMA 50/200 vs. ML (XGBoost)")
    print("=" * len(header))
    print(header)
    print(sep)

    for ticker in config.WATCHLIST:
        sma = sma_results.get(ticker)
        ml = ml_results.get(ticker)

        if sma is not None:
            print(
                f"{ticker:<10}{'SMA 50/200':<16}"
                f"{_fmt_pct(sma.get('total_return_pct')):>9}"
                f"{_fmt_pct(sma.get('buy_hold_return_pct')):>9}"
                f"{sma.get('n_trades', 'n/a'):>8}"
                f"{_fmt_pct((sma.get('win_rate') or 0) * 100):>8}"
                f"{_fmt_num(sma.get('sharpe_ratio')):>8}"
                f"{_fmt_pct(sma.get('max_drawdown_pct')):>8}"
            )
        else:
            print(f"{ticker:<10}{'SMA 50/200':<16}{'no data':>9}")

        if ml is not None:
            ml_win_rate = ml.get("win_rate")
            # win_rate may already be stored as a 0-100 percentage or a 0-1
            # fraction depending on how the ML backtester wrote it; normalize.
            if ml_win_rate is not None and ml_win_rate <= 1.0:
                ml_win_rate = ml_win_rate * 100
            print(
                f"{ticker:<10}{'ML (XGBoost)':<16}"
                f"{_fmt_pct(ml.get('total_return_pct')):>9}"
                f"{_fmt_pct(ml.get('buy_hold_return_pct')):>9}"
                f"{ml.get('n_trades', 'n/a'):>8}"
                f"{_fmt_pct(ml_win_rate):>8}"
                f"{_fmt_num(ml.get('sharpe_ratio')):>8}"
                f"{_fmt_pct(ml.get('max_drawdown_pct')):>8}"
            )
        else:
            print(f"{ticker:<10}{'ML (XGBoost)':<16}{'no data':>9}")

        print(sep)

    print()


def print_verdict(sma_results: dict, ml_results: dict) -> None:
    """Print the head-to-head summary and final verdict."""
    sma_beats_ml = 0
    ml_beats_sma = 0
    sma_beats_bh = 0
    ml_beats_bh = 0
    n_compared = 0

    for ticker in config.WATCHLIST:
        sma = sma_results.get(ticker)
        ml = ml_results.get(ticker)

        if sma is not None and sma.get("total_return_pct") is not None:
            if sma["total_return_pct"] > sma.get("buy_hold_return_pct", float("inf")):
                sma_beats_bh += 1

        if ml is not None and ml.get("total_return_pct") is not None:
            if ml["total_return_pct"] > ml.get("buy_hold_return_pct", float("inf")):
                ml_beats_bh += 1

        if sma is not None and ml is not None:
            n_compared += 1
            if sma.get("total_return_pct", float("-inf")) > ml.get("total_return_pct", float("-inf")):
                sma_beats_ml += 1
            elif ml.get("total_return_pct", float("-inf")) > sma.get("total_return_pct", float("-inf")):
                ml_beats_sma += 1

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tickers compared head-to-head: {n_compared}")
    print(f"  SMA 50/200 beat ML:        {sma_beats_ml}")
    print(f"  ML beat SMA 50/200:        {ml_beats_sma}")
    print(f"  SMA 50/200 beat buy&hold:  {sma_beats_bh}")
    print(f"  ML beat buy&hold:          {ml_beats_bh}")
    print("=" * 60)

    if ml_beats_sma > sma_beats_ml:
        print("✅ ML BEATS SMA. The complexity is paying off.")
    elif sma_beats_ml > ml_beats_sma:
        print("🚨 SMA BEATS ML. The ML isn't adding value yet.")
    else:
        print("🤷 TIE. ML complexity isn't justified.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    sma_results = load_sma_results()
    ml_results = load_ml_results()

    if not sma_results:
        logger.warning("No SMA results loaded — run sma_baseline.py first.")
    if not ml_results:
        logger.warning("No ML results loaded — check backtest_results table in stockoracle.db.")

    print_comparison(sma_results, ml_results)
    print_verdict(sma_results, ml_results)
