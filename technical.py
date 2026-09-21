"""
technical.py — technical indicators computed from stored price history.

Uses the `ta` library (pure pandas/numpy under the hood, no C extensions)
per the project's tech stack. Indicators computed:
  - RSI(14)
  - MACD line + signal line (12/26/9, ta's defaults)
  - Bollinger Bands (20-period, 2 std dev) upper/lower
  - SMA(50), SMA(200)

Design notes:
  - Indicators are computed over the *entire* stored price history for a
    ticker, not just the latest bar, and every date's row is upserted into
    `technical_indicators`. This matters for Phase 3's backtester: a
    walk-forward backtest needs the indicator value *as it would have been
    known* on each historical date, not just today's snapshot.
  - Early dates (fewer than 200 bars of history) will have NULL sma_200,
    NULL bb_*, etc. wherever `ta` can't yet compute a value — this is
    correct and expected, not a bug. Callers (predictor.py, decision_engine.py)
    must handle NULLs rather than assume every row is fully populated.
  - Like data_fetcher.py, this module never crashes on a missing dependency:
    if `ta` or `pandas` aren't installed, it logs a warning and returns
    without writing anything.
  - No look-ahead bias: every indicator here is computed using only that
    row's date and prior rows (standard trailing technical-indicator math).
    Nothing here peeks at future price bars.
"""

from __future__ import annotations

import sqlite3

import config
import database

log = config.get_logger(__name__)

# Minimum number of price bars required before we even attempt indicator
# calculation. Below this, RSI/MACD/Bollinger are too noisy to be meaningful.
MIN_BARS_REQUIRED = 20


def _compute_with_ta_library(df, ta_module) -> None:
    """Populate indicator columns on `df` using the `ta` library. Mutates df in place."""
    df["rsi_14"] = ta_module.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    macd_calc = ta_module.trend.MACD(close=df["close"])
    df["macd"] = macd_calc.macd()
    df["macd_signal"] = macd_calc.macd_signal()

    bb = ta_module.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    df["sma_50"] = ta_module.trend.SMAIndicator(close=df["close"], window=50).sma_indicator()
    df["sma_200"] = ta_module.trend.SMAIndicator(close=df["close"], window=200).sma_indicator()


def _compute_with_pandas_fallback(df) -> None:
    """
    Populate indicator columns on `df` using plain pandas math, for
    environments where the `ta` package isn't installed. Formulas match
    `ta`'s defaults (Wilder's smoothing for RSI, EMA 12/26/9 for MACD,
    20-period/2-std Bollinger Bands) closely enough for research purposes.
    Mutates df in place. Trailing-only — no look-ahead.
    """
    close = df["close"]

    # RSI(14), Wilder's smoothing (matches `ta`'s RSIIndicator).
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df.loc[avg_loss == 0, "rsi_14"] = 100.0

    # MACD(12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    df["macd"] = macd_line
    df["macd_signal"] = macd_line.ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2 std)
    sma_20 = close.rolling(window=20, min_periods=20).mean()
    std_20 = close.rolling(window=20, min_periods=20).std(ddof=0)
    df["bb_upper"] = sma_20 + 2 * std_20
    df["bb_lower"] = sma_20 - 2 * std_20

    # SMAs
    df["sma_50"] = close.rolling(window=50, min_periods=50).mean()
    df["sma_200"] = close.rolling(window=200, min_periods=200).mean()


def compute_indicators_for_ticker(conn: sqlite3.Connection, ticker: str) -> int:
    """
    Load all stored price history for `ticker`, compute indicators for every
    date, and upsert them into `technical_indicators`. Returns the number of
    dates written (0 if there wasn't enough data or pandas was missing).

    Uses the `ta` library if installed; otherwise falls back to an
    equivalent pure-pandas implementation so this module works even in a
    fresh environment before `pip install ta` has been run.
    """
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not installed — skipping technical indicators for %s.", ticker)
        return 0

    rows = database.get_price_history(conn, ticker)
    if len(rows) < MIN_BARS_REQUIRED:
        log.warning(
            "Only %d price bars stored for %s (need >= %d) — skipping indicators.",
            len(rows), ticker, MIN_BARS_REQUIRED,
        )
        return 0

    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    try:
        import ta as ta_module
        _compute_with_ta_library(df, ta_module)
    except ImportError:
        log.warning("`ta` not installed for %s — using built-in pandas fallback formulas.", ticker)
        _compute_with_pandas_fallback(df)
    except Exception as exc:
        log.error("Failed computing indicators for %s via `ta`: %s", ticker, exc)
        return 0

    # pandas/NaN -> Python None so sqlite stores real NULLs, not the string "nan"
    df = df.where(pd.notna(df), None)

    written = 0
    for _, row in df.iterrows():
        database.upsert_technical_indicators(
            conn,
            ticker=ticker,
            date=row["date"].strftime("%Y-%m-%d"),
            rsi_14=row["rsi_14"],
            macd=row["macd"],
            macd_signal=row["macd_signal"],
            bb_upper=row["bb_upper"],
            bb_lower=row["bb_lower"],
            sma_50=row["sma_50"],
            sma_200=row["sma_200"],
        )
        written += 1

    log.info("Computed and stored indicators for %s across %d dates.", ticker, written)
    return written


def compute_indicators_for_watchlist(conn: sqlite3.Connection, tickers: list[str] | None = None) -> dict[str, int]:
    """Run compute_indicators_for_ticker for every ticker in the watchlist (or
    the given list). One ticker's failure doesn't stop the others."""
    tickers = tickers or config.WATCHLIST
    results: dict[str, int] = {}
    for ticker in tickers:
        try:
            results[ticker] = compute_indicators_for_ticker(conn, ticker)
        except Exception as exc:
            log.error("Unexpected error computing indicators for %s: %s", ticker, exc)
            results[ticker] = 0
    return results


if __name__ == "__main__":
    database.init_db()
    with database.get_connection() as conn:
        summary = compute_indicators_for_watchlist(conn)
        log.info("=== Technical indicator summary ===")
        for ticker, count in summary.items():
            log.info("%s: %d dates written", ticker, count)
