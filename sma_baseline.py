"""
sma_baseline.py

Implements a classic 50/200-day SMA crossover ("Golden Cross / Death Cross")
backtest for every ticker in config.WATCHLIST, using the same price data,
date range, and transaction cost assumptions as the ML backtest, so the two
are directly comparable.

Strategy logic (long/flat, no shorting):
    - Start in cash.
    - When SMA(50) crosses above SMA(200) and we are in cash -> buy at the
      *next* bar's open (avoids look-ahead bias: the crossover is only known
      after today's close, so the earliest we can act is tomorrow's open).
    - When SMA(50) crosses below SMA(200) and we are long -> sell at the
      next bar's open.
    - Transaction costs (config.TRANSACTION_COST_PCT) are applied on both
      the buy and the sell, modeled as slippage on the execution price.
    - Any open position is marked-to-market daily and force-closed at the
      end of the test window (at the final close) so total_return_pct
      reflects a fully realized result.

Results are written to backtests/sma_baseline_results.json as
{ticker: metrics_dict}, and a summary table is printed to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from database import get_connection, get_price_history

logger = config.get_logger(__name__)

SMA_FAST = 50
SMA_SLOW = 200
TRADING_DAYS_PER_YEAR = 252
OUTPUT_PATH = Path("backtests") / "sma_baseline_results.json"


def _max_drawdown_pct(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough decline in an equity curve, as a positive percentage."""
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min() * 100.0)


def _sharpe_ratio(daily_returns: pd.Series) -> float:
    """Annualized Sharpe ratio from a series of daily equity returns (rf = 0)."""
    if daily_returns.empty or daily_returns.std(ddof=0) == 0 or daily_returns.isna().all():
        return 0.0
    mean = daily_returns.mean()
    std = daily_returns.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * mean / std)


def _run_single_ticker(ticker: str, df: pd.DataFrame) -> dict | None:
    """Simulate the SMA 50/200 crossover strategy for one ticker's OHLCV history."""
    if df is None or df.empty:
        logger.warning("No price history for %s, skipping.", ticker)
        return None

    df = df.sort_values("date").reset_index(drop=True).copy()

    required_cols = {"date", "open", "close"}
    if not required_cols.issubset(df.columns):
        logger.warning(
            "Price history for %s missing required columns %s, skipping.",
            ticker,
            required_cols - set(df.columns),
        )
        return None

    df["sma_fast"] = df["close"].rolling(window=SMA_FAST, min_periods=SMA_FAST).mean()
    df["sma_slow"] = df["close"].rolling(window=SMA_SLOW, min_periods=SMA_SLOW).mean()

    # Only consider bars where both SMAs are defined (i.e. after bar 200).
    tradeable = df[df["sma_slow"].notna()].copy()
    if tradeable.empty or len(tradeable) < 2:
        logger.warning(
            "Not enough history for %s to compute SMA(%d)/SMA(%d) crossovers, skipping.",
            ticker,
            SMA_FAST,
            SMA_SLOW,
        )
        return None

    tradeable = tradeable.reset_index(drop=True)
    tradeable["signal"] = np.where(tradeable["sma_fast"] > tradeable["sma_slow"], 1, 0)
    # Crossover happens on the bar where the signal changes vs. the prior bar.
    tradeable["signal_prev"] = tradeable["signal"].shift(1)

    cost = config.TRANSACTION_COST_PCT

    position = 0  # 0 = cash, 1 = long
    shares = 0.0
    cash = 1.0  # start with $1 of capital
    entry_price = None
    trades: list[dict] = []
    equity_curve: list[float] = []
    dates: list = []
    invested_days = 0
    n_tradeable_bars = len(tradeable)

    for i in range(n_tradeable_bars):
        row = tradeable.iloc[i]
        date = row["date"]

        # Detect crossover using prior bar's signal vs. current bar's signal.
        # We only know the crossover as of *this* bar's close, so execution
        # happens at the *next* bar's open (i + 1), never at this bar's own
        # open or close -> no look-ahead bias.
        if i > 0 and not pd.isna(row["signal_prev"]):
            golden_cross = row["signal"] == 1 and row["signal_prev"] == 0
            death_cross = row["signal"] == 0 and row["signal_prev"] == 1

            if golden_cross and position == 0 and i + 1 < n_tradeable_bars:
                next_open = tradeable.iloc[i + 1]["open"]
                exec_price = next_open * (1 + cost)
                shares = cash / exec_price
                cash = 0.0
                position = 1
                entry_price = exec_price
                entry_date = tradeable.iloc[i + 1]["date"]

            elif death_cross and position == 1 and i + 1 < n_tradeable_bars:
                next_open = tradeable.iloc[i + 1]["open"]
                exec_price = next_open * (1 - cost)
                cash = shares * exec_price
                trade_return = (exec_price - entry_price) / entry_price
                trades.append(
                    {
                        "entry_date": str(entry_date),
                        "exit_date": str(tradeable.iloc[i + 1]["date"]),
                        "entry_price": float(entry_price),
                        "exit_price": float(exec_price),
                        "return_pct": float(trade_return * 100.0),
                    }
                )
                shares = 0.0
                position = 0
                entry_price = None

        # Mark-to-market equity for this bar's close.
        if position == 1:
            mtm_equity = shares * row["close"]
            invested_days += 1
        else:
            mtm_equity = cash
        equity_curve.append(mtm_equity)
        dates.append(date)

    # Force-close any open position at the final close so returns are fully realized.
    if position == 1:
        final_close = tradeable.iloc[-1]["close"]
        exec_price = final_close * (1 - cost)
        cash = shares * exec_price
        trade_return = (exec_price - entry_price) / entry_price
        trades.append(
            {
                "entry_date": str(entry_date),
                "exit_date": str(tradeable.iloc[-1]["date"]) + " (forced close, end of window)",
                "entry_price": float(entry_price),
                "exit_price": float(exec_price),
                "return_pct": float(trade_return * 100.0),
            }
        )
        equity_curve[-1] = cash
        position = 0
        shares = 0.0

    equity_series = pd.Series(equity_curve, index=pd.to_datetime(dates))
    daily_returns = equity_series.pct_change().dropna()

    n_trades = len(trades)
    win_rate = (
        float(sum(1 for t in trades if t["return_pct"] > 0) / n_trades) if n_trades > 0 else 0.0
    )
    total_return_pct = float((equity_series.iloc[-1] - 1.0) * 100.0) if not equity_series.empty else 0.0

    first_close = tradeable.iloc[0]["close"]
    last_close = tradeable.iloc[-1]["close"]
    buy_hold_return_pct = float((last_close - first_close) / first_close * 100.0)

    max_dd = _max_drawdown_pct(equity_series)
    sharpe = _sharpe_ratio(daily_returns)
    exposure_pct = float(invested_days / n_tradeable_bars * 100.0) if n_tradeable_bars > 0 else 0.0

    metrics = {
        "ticker": ticker,
        "strategy": "SMA_50_200_CROSSOVER",
        "start_date": str(tradeable.iloc[0]["date"]),
        "end_date": str(tradeable.iloc[-1]["date"]),
        "n_bars": int(n_tradeable_bars),
        "n_trades": int(n_trades),
        "win_rate": win_rate,
        "total_return_pct": total_return_pct,
        "buy_hold_return_pct": buy_hold_return_pct,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "exposure_pct": exposure_pct,
        "trades": trades,
    }
    return metrics


def run_sma_baseline() -> dict:
    """Run the SMA crossover backtest for every ticker in the watchlist."""
    results: dict[str, dict] = {}

    with get_connection() as conn:
        for ticker in config.WATCHLIST:
            logger.info("Running SMA(%d/%d) baseline for %s...", SMA_FAST, SMA_SLOW, ticker)
            try:
                df = get_price_history(conn, ticker)
            except Exception:
                logger.exception("Failed to load price history for %s", ticker)
                continue

            metrics = _run_single_ticker(ticker, df)
            if metrics is not None:
                results[ticker] = metrics

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved SMA baseline results to %s", OUTPUT_PATH)

    return results


def print_summary(results: dict) -> None:
    """Print a summary table of the SMA baseline results."""
    header = (
        f"{'Ticker':<8}{'Return':>10}{'B&H':>10}{'Trades':>8}"
        f"{'Win%':>8}{'Sharpe':>9}{'MaxDD':>9}{'Exposure':>10}"
    )
    print("\n" + "=" * len(header))
    print("SMA 50/200 CROSSOVER BASELINE — SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for ticker, m in results.items():
        print(
            f"{ticker:<8}"
            f"{m['total_return_pct']:>9.1f}%"
            f"{m['buy_hold_return_pct']:>9.1f}%"
            f"{m['n_trades']:>8}"
            f"{m['win_rate'] * 100:>7.1f}%"
            f"{m['sharpe_ratio']:>9.2f}"
            f"{m['max_drawdown_pct']:>8.1f}%"
            f"{m['exposure_pct']:>9.1f}%"
        )
    print("=" * len(header) + "\n")


if __name__ == "__main__":
    results = run_sma_baseline()
    print_summary(results)
