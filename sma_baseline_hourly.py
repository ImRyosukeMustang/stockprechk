from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
import database

log = config.get_logger(__name__)

FAST_WINDOW = 20
SLOW_WINDOW = 50
RESULTS_PATH = Path("backtests/sma_hourly_results.json")


def fetch_hourly_data(ticker: str) -> pd.DataFrame:
    """Fetch hourly price bars from the database for a given ticker."""
    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM prices_intraday
        WHERE ticker = ? AND interval = '1h'
        ORDER BY timestamp ASC
    """
    try:
        with database.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(ticker,))
    except Exception as exc:
        log.error("Failed to query database for ticker %s: %s", ticker, exc)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def run_hourly_sma_backtest(df: pd.DataFrame, ticker: str) -> dict[str, Any] | None:
    """Simulate a long/cash SMA(20/50) strategy on hourly prices."""
    if len(df) < SLOW_WINDOW + 1:
        log.warning(
            "Insufficient hourly data for %s (%d bars available, need at least %d). Skipping.",
            ticker,
            len(df),
            SLOW_WINDOW + 1,
        )
        return None

    cost_pct = getattr(config, "TRANSACTION_COST_PCT", 0.001)

    df = df.copy()
    df["sma_fast"] = df["close"].rolling(window=FAST_WINDOW).mean()
    df["sma_slow"] = df["close"].rolling(window=SLOW_WINDOW).mean()
    df["signal"] = 0
    df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
    df["position"] = df["signal"].shift(1).fillna(0).astype(int)

    n_bars = len(df)
    in_position = False
    entry_price = 0.0
    trades: list[dict[str, Any]] = []
    position_mask = np.zeros(n_bars)

    for i in range(SLOW_WINDOW, n_bars):
        curr_pos = df["position"].iloc[i]
        prev_pos = df["position"].iloc[i - 1]
        curr_price = df["close"].iloc[i]

        if curr_pos == 1 and prev_pos == 0:
            in_position = True
            entry_price = curr_price * (1 + cost_pct)
        elif curr_pos == 0 and prev_pos == 1 and in_position:
            exit_price = curr_price * (1 - cost_pct)
            ret = (exit_price - entry_price) / entry_price
            trades.append({"entry": entry_price, "exit": exit_price, "return": ret})
            in_position = False

        if in_position:
            position_mask[i] = 1

    if in_position:
        last_price = df["close"].iloc[-1] * (1 - cost_pct)
        ret = (last_price - entry_price) / entry_price
        trades.append({"entry": entry_price, "exit": last_price, "return": ret})
        position_mask[-1] = 1

    df["asset_return"] = df["close"].pct_change().fillna(0.0)
    strat_returns = df["position"] * df["asset_return"]
    trade_signals = df["position"].diff().abs().fillna(0.0)
    strat_returns -= trade_signals * cost_pct
    df["equity_curve"] = (1 + strat_returns).cumprod()

    total_return_pct = float((df["equity_curve"].iloc[-1] - 1.0) * 100)
    buy_hold_return_pct = float(((df["close"].iloc[-1] / df["close"].iloc[0]) - 1.0) * 100)

    n_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade["return"] > 0)
    win_rate = float((winning_trades / n_trades * 100) if n_trades > 0 else 0.0)

    mean_ret = strat_returns.mean()
    std_ret = strat_returns.std()
    hourly_sharpe = float((mean_ret / std_ret) * np.sqrt(1764)) if std_ret > 0 else 0.0

    running_max = df["equity_curve"].cummax()
    drawdown = (df["equity_curve"] - running_max) / running_max
    max_drawdown_pct = float(drawdown.min() * 100)
    exposure_pct = float((position_mask.sum() / n_bars) * 100)

    return {
        "ticker": ticker,
        "total_return_pct": round(total_return_pct, 2),
        "buy_hold_return_pct": round(buy_hold_return_pct, 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 2),
        "sharpe_ratio": round(hourly_sharpe, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "exposure_pct": round(exposure_pct, 2),
    }


def main() -> None:
    watchlist = getattr(config, "WATCHLIST", ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"])
    results: list[dict[str, Any]] = []

    log.info("Starting Hourly SMA (20/50) Backtest...")

    for ticker in watchlist:
        log.info("Processing %s...", ticker)
        df = fetch_hourly_data(ticker)
        if df.empty:
            log.warning("No intraday data found for %s. Skipping.", ticker)
            continue

        result = run_hourly_sma_backtest(df, ticker)
        if result:
            results.append(result)

    if not results:
        log.error("No valid results generated for any ticker.")
        return

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)
    log.info("Results successfully saved to %s", RESULTS_PATH)

    print("\n" + "=" * 85)
    print(f"{'HOURLY SMA (20/50) BACKTEST RESULTS':^85}")
    print("=" * 85)
    print(
        f"{'Ticker':<8} {'Return %':<10} {'B&H %':<10} {'Trades':<8} "
        f"{'Win %':<8} {'Sharpe':<8} {'Max DD %':<10} {'Exposure %':<10}"
    )
    print("-" * 85)
    for result in results:
        print(
            f"{result['ticker']:<8} {result['total_return_pct']:<10.2f} {result['buy_hold_return_pct']:<10.2f} "
            f"{result['n_trades']:<8} {result['win_rate']:<8.2f} {result['sharpe_ratio']:<8.2f} "
            f"{result['max_drawdown_pct']:<10.2f} {result['exposure_pct']:<10.2f}"
        )
    print("=" * 85)


if __name__ == "__main__":
    main()
