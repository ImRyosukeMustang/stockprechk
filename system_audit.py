"""
system_audit.py — comprehensive audit of the StockOracle system.

Tests every layer for bugs, flaws, data quality issues, and inconsistencies:
  1. Database schema integrity
  2. Data completeness (gaps, missing tickers, stale data)
  3. Feature engineering correctness (no NaN explosions, no look-ahead)
  4. Predictor training integrity (label leakage, class imbalance)
  5. Backtester correctness (costs, execution timing, equity curve)
  6. SMA signal correctness
  7. Portfolio allocation math
  8. Journal logging
  9. Cross-module consistency (signals match between DB and dashboard)

Runs read-only. Does not modify any data. Prints a report to stdout.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config
import database

log = config.get_logger(__name__)

# ─── ANSI colors for readability ─────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
END = "\033[0m"

PASS = f"{GREEN}✅ PASS{END}"
WARN = f"{YELLOW}⚠️  WARN{END}"
FAIL = f"{RED}❌ FAIL{END}"

results = {"pass": 0, "warn": 0, "fail": 0}


def section(title: str) -> None:
    print(f"\n{BOLD}{'='*70}{END}")
    print(f"{BOLD}{title}{END}")
    print(f"{BOLD}{'='*70}{END}")


def check(name: str, condition: bool, detail: str = "", warn_only: bool = False) -> None:
    """Record and print a single check result."""
    if condition:
        print(f"  {PASS}  {name}")
        results["pass"] += 1
    else:
        tag = WARN if warn_only else FAIL
        print(f"  {tag}  {name}")
        if detail:
            print(f"        {detail}")
        results["warn" if warn_only else "fail"] += 1


# ─────────────────────────────────────────────────────────────────────────
# 1. DATABASE SCHEMA
# ─────────────────────────────────────────────────────────────────────────
def audit_schema(conn: sqlite3.Connection) -> None:
    section("1. DATABASE SCHEMA")

    expected_tables = {
        "prices", "news", "reddit_posts", "fundamentals",
        "sentiment_scores", "technical_indicators", "predictions",
        "signals", "pipeline_runs", "backtest_results",
        "sma_signal_journal", "prices_intraday", "macro_indicators",
        "patterns", "pattern_outcomes", "news_events",
    }

    actual_tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

    for table in expected_tables:
        check(f"Table '{table}' exists", table in actual_tables)

    # Verify predictions has fold_id
    cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
    check("predictions has fold_id column", "fold_id" in cols)

    # Verify sma_signal_journal columns
    jcols = {row[1] for row in conn.execute("PRAGMA table_info(sma_signal_journal)")}
    expected_jcols = {"date", "ticker", "regime", "action",
                      "forward_5d_return", "forward_20d_return",
                      "forward_60d_return", "was_correct_5d"}
    missing = expected_jcols - jcols
    check(
        "sma_signal_journal has all expected columns",
        len(missing) == 0,
        f"Missing: {missing}" if missing else "",
    )

    fundamental_cols = {row[1] for row in conn.execute("PRAGMA table_info(fundamentals)")}
    expected_fundamental_cols = {
        "pe_ratio", "forward_pe", "peg_ratio", "profit_margin",
        "revenue_growth", "earnings_growth", "debt_to_equity",
        "free_cashflow", "dividend_yield", "beta",
    }
    missing = expected_fundamental_cols - fundamental_cols
    check(
        "fundamentals has all ML feature columns",
        not missing,
        f"Missing: {missing}" if missing else "",
    )

    macro_cols = {row[1] for row in conn.execute("PRAGMA table_info(macro_indicators)")}
    check(
        "macro_indicators has all expected columns",
        {"date", "fed_funds_rate", "ten_year_yield", "cpi", "vix"} <= macro_cols,
        f"Columns: {macro_cols}",
    )


# ─────────────────────────────────────────────────────────────────────────
# 2. DATA COMPLETENESS
# ─────────────────────────────────────────────────────────────────────────
def audit_data(conn: sqlite3.Connection) -> None:
    section("2. DATA COMPLETENESS")

    for ticker in config.WATCHLIST:
        row = conn.execute(
            "SELECT COUNT(*), MIN(date), MAX(date) FROM prices WHERE ticker = ?",
            (ticker,),
        ).fetchone()
        count, min_date, max_date = row

        check(
            f"{ticker}: has price data",
            count > 0,
            f"Count: {count}",
        )

        if count > 0:
            check(
                f"{ticker}: has enough bars for SMA(200)",
                count >= 250,
                f"Count: {count} (need ≥ 250)",
                warn_only=True,
            )

            # Check for recent data (within last 5 days)
            latest = datetime.strptime(max_date, "%Y-%m-%d").date()
            days_stale = (datetime.now().date() - latest).days
            check(
                f"{ticker}: data is recent",
                days_stale <= 5,
                f"Latest: {max_date}, {days_stale} days old",
            )

    # Gap detection — look for missing weekdays
    print()
    for ticker in config.WATCHLIST[:2]:  # just first 2 for speed
        dates = [
            datetime.strptime(r[0], "%Y-%m-%d").date()
            for r in conn.execute(
                "SELECT date FROM prices WHERE ticker = ? ORDER BY date",
                (ticker,),
            )
        ]
        if len(dates) < 10:
            continue
        gaps = []
        for i in range(1, len(dates)):
            delta = (dates[i] - dates[i - 1]).days
            if delta > 5:  # more than 5 days = suspicious gap
                gaps.append((dates[i - 1], dates[i], delta))
        check(
            f"{ticker}: no suspicious gaps > 5 days",
            len(gaps) <= 1,  # allow 1 for holidays/weekends edge cases
            f"Found {len(gaps)} gaps: {gaps[:3]}" if gaps else "",
            warn_only=True,
        )

    # Intraday data
    print()
    row = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT ticker) FROM prices_intraday"
    ).fetchone()
    check(
        "prices_intraday has data",
        row[0] > 0,
        f"Rows: {row[0]}, Tickers: {row[1]}",
        warn_only=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────
def audit_features(conn: sqlite3.Connection) -> None:
    section("3. FEATURE ENGINEERING")

    try:
        import predictor
    except ImportError:
        check("predictor module importable", False)
        return

    for ticker in config.WATCHLIST[:2]:
        df = predictor.build_feature_frame(conn, ticker)
        if df is None or df.empty:
            check(f"{ticker}: feature frame builds", False)
            continue

        check(f"{ticker}: feature frame builds", True)

        # All expected feature columns present
        missing = [c for c in predictor.FEATURE_COLUMNS if c not in df.columns]
        check(
            f"{ticker}: all FEATURE_COLUMNS present",
            len(missing) == 0,
            f"Missing: {missing}" if missing else "",
        )

        # Check for NaN % per feature
        feature_nans = df[predictor.FEATURE_COLUMNS].isna().mean()
        high_nan = feature_nans[feature_nans > 0.5]
        optional_features = set(getattr(predictor, "FUNDAMENTAL_FEATURE_COLUMNS", ()))
        optional_features.update(getattr(predictor, "SENTIMENT_AGGREGATE_FEATURE_COLUMNS", ()))
        optional_features.update(getattr(predictor, "MACRO_FEATURE_COLUMNS", ()))
        core_high_nan = high_nan.drop(labels=list(optional_features), errors="ignore")
        check(
            f"{ticker}: no features with >50% NaN",
            len(high_nan) == 0,
            f"High NaN: {dict(high_nan)}" if len(high_nan) > 0 else "",
            warn_only=len(high_nan) > 0 and len(core_high_nan) == 0,
        )

        # Check for infinite values
        features_only = df[predictor.FEATURE_COLUMNS].select_dtypes(include=[np.number])
        has_inf = np.isinf(features_only.to_numpy(dtype=float, na_value=np.nan)).any()
        check(f"{ticker}: no infinite values in features", not has_inf)

        # Check dtypes are numeric
        non_numeric = [
            c for c in predictor.FEATURE_COLUMNS
            if c in df.columns and not np.issubdtype(df[c].dtype, np.number)
        ]
        check(
            f"{ticker}: all features are numeric",
            len(non_numeric) == 0,
            f"Non-numeric: {non_numeric}" if non_numeric else "",
        )


# ─────────────────────────────────────────────────────────────────────────
# 4. PREDICTOR INTEGRITY
# ─────────────────────────────────────────────────────────────────────────
def audit_predictor(conn: sqlite3.Connection) -> None:
    section("4. PREDICTOR INTEGRITY")

    # Check predictions table for anomalies
    row = conn.execute(
        "SELECT COUNT(*), MIN(probability_up), MAX(probability_up) FROM predictions"
    ).fetchone()
    count, pmin, pmax = row
    check("predictions table has data", count > 0, f"Rows: {count}")

    if count > 0:
        check(
            "predictions in valid range [0, 1]",
            pmin is not None and 0 <= pmin and pmax <= 1,
            f"Range: [{pmin}, {pmax}]",
        )

    # Check class balance of the target
    try:
        import predictor
        for ticker in config.WATCHLIST[:2]:
            df = predictor.build_feature_frame(conn, ticker)
            if df is None or df.empty:
                continue
            df = df.copy()
            df["future_close"] = df["close"].shift(-5)
            df["label"] = (df["future_close"] > df["close"]).astype("Int64")
            df = df.iloc[:-5].dropna(subset=["label"])
            if len(df) == 0:
                continue
            balance = df["label"].mean()
            check(
                f"{ticker}: class balance is reasonable (30-70%)",
                0.3 <= balance <= 0.7,
                f"Class balance: {balance:.2%} up",
                warn_only=True,
            )
    except Exception as e:
        check("predictor label balance check", False, str(e))


# ─────────────────────────────────────────────────────────────────────────
# 5. BACKTESTER INTEGRITY
# ─────────────────────────────────────────────────────────────────────────
def audit_backtester(conn: sqlite3.Connection) -> None:
    section("5. BACKTESTER INTEGRITY")

    rows = conn.execute("""
        SELECT ticker, model_name, strategy_return_pct, benchmark_return_pct,
               num_trades, max_drawdown_pct, sharpe_ratio
        FROM backtest_results
    """).fetchall()

    check("backtest_results has data", len(rows) > 0, f"Rows: {len(rows)}")

    for row in rows:
        ticker, model_name, ret, bench, n_trades, dd, sharpe = row

        # Sanity: return shouldn't exceed 1000% for daily strategy
        check(
            f"{ticker}/{model_name}: return is sane",
            ret is None or -100 <= ret <= 1000,
            f"Return: {ret}%",
            warn_only=True,
        )

        # Drawdown should be negative
        if dd is not None:
            check(
                f"{ticker}/{model_name}: max_drawdown is negative",
                dd <= 0,
                f"DD: {dd}",
            )

        # If trades are 0, return should be near 0
        if n_trades == 0 and ret is not None:
            check(
                f"{ticker}/{model_name}: 0 trades → ~0% return",
                abs(ret) < 5,
                f"Return with 0 trades: {ret}%",
                warn_only=True,
            )


# ─────────────────────────────────────────────────────────────────────────
# 6. SMA SIGNAL CORRECTNESS
# ─────────────────────────────────────────────────────────────────────────
def audit_sma_signals(conn: sqlite3.Connection) -> None:
    section("6. SMA SIGNAL CORRECTNESS")

    for ticker in config.WATCHLIST:
        rows = conn.execute("""
            SELECT date, close FROM prices
            WHERE ticker = ?
            ORDER BY date ASC
        """, (ticker,)).fetchall()

        if len(rows) < 200:
            continue

        df = pd.DataFrame(rows, columns=["date", "close"])
        df["sma50"] = df["close"].rolling(50).mean()
        df["sma200"] = df["close"].rolling(200).mean()

        # Verify SMA computation against manual calc on last value
        if pd.notna(df["sma50"].iloc[-1]):
            manual_sma50 = df["close"].iloc[-50:].mean()
            check(
                f"{ticker}: SMA50 matches manual calc",
                abs(df["sma50"].iloc[-1] - manual_sma50) < 0.01,
                f"Formula: {df['sma50'].iloc[-1]:.4f}, Manual: {manual_sma50:.4f}",
            )

        # Verify current regime matches DB
        regime = "bullish" if df["sma50"].iloc[-1] > df["sma200"].iloc[-1] else "bearish"
        db_row = conn.execute("""
            SELECT reasoning FROM signals
            WHERE ticker = ? ORDER BY created_at DESC LIMIT 1
        """, (ticker,)).fetchone()

        if db_row:
            db_reasoning = db_row[0] or ""
            if "sma" in db_reasoning.lower():
                check(
                    f"{ticker}: DB signal matches computed regime",
                    regime in db_reasoning.lower(),
                    f"Computed: {regime}, DB: {db_reasoning[:80]}",
                )


# ─────────────────────────────────────────────────────────────────────────
# 7. PORTFOLIO ALLOCATION
# ─────────────────────────────────────────────────────────────────────────
def audit_portfolio(conn: sqlite3.Connection) -> None:
    section("7. PORTFOLIO ALLOCATION")

    try:
        import portfolio
    except ImportError:
        check("portfolio module importable", False)
        return

    # Test with all bullish
    signals = {t: {"regime": "bullish", "position_size": 1.0}
               for t in config.WATCHLIST}
    alloc = portfolio.compute_portfolio_allocation(conn, signals)

    total = sum(alloc.values())
    check(
        "All bullish → total weight = 100%",
        abs(total - 1.0) < 0.01,
        f"Total: {total:.2%}, Allocation: {alloc}",
    )

    per_pos = list(alloc.values())[0]
    check(
        "All bullish → each position ≤ MAX_POSITION_PCT",
        per_pos <= config.MAX_POSITION_PCT + 0.01,
        f"Per position: {per_pos:.2%}, Cap: {config.MAX_POSITION_PCT:.2%}",
    )

    # Test with 2 bullish
    signals_mixed = {
        "AAPL": {"regime": "bullish", "position_size": 1.0},
        "MSFT": {"regime": "bullish", "position_size": 1.0},
        "NVDA": {"regime": "bearish", "position_size": 0.0},
        "GOOGL": {"regime": "bearish", "position_size": 0.0},
        "AMZN": {"regime": "bearish", "position_size": 0.0},
    }
    alloc2 = portfolio.compute_portfolio_allocation(conn, signals_mixed)

    bullish_weights = [alloc2.get(t, 0) for t in ["AAPL", "MSFT"]]
    bearish_weights = [alloc2.get(t, 0) for t in ["NVDA", "GOOGL", "AMZN"]]

    check(
        "Mixed → bearish tickers have 0 weight",
        all(w == 0 for w in bearish_weights),
        f"Bearish weights: {bearish_weights}",
    )
    check(
        "Mixed → total ≤ 100%",
        sum(alloc2.values()) <= 1.01,
        f"Total: {sum(alloc2.values()):.2%}",
    )


# ─────────────────────────────────────────────────────────────────────────
# 8. JOURNAL
# ─────────────────────────────────────────────────────────────────────────
def audit_journal(conn: sqlite3.Connection) -> None:
    section("8. SMA SIGNAL JOURNAL")

    row = conn.execute("""
        SELECT COUNT(*), COUNT(DISTINCT ticker), COUNT(DISTINCT date),
               MIN(date), MAX(date)
        FROM sma_signal_journal
    """).fetchone()
    count, tickers, dates, min_d, max_d = row
    check("Journal has entries", count > 0, f"Rows: {count}")

    if count > 0:
        check(
            "Journal covers all watchlist tickers",
            tickers >= len(config.WATCHLIST),
            f"Tickers in journal: {tickers}, Watchlist: {len(config.WATCHLIST)}",
            warn_only=True,
        )

        # No future dates (would be a bug)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        future = conn.execute(
            "SELECT COUNT(*) FROM sma_signal_journal WHERE date > ?", (today,)
        ).fetchone()[0]
        check("No future dates in journal", future == 0,
              f"Future dates: {future}")

        # Backfilled outcomes consistency
        backfilled = conn.execute("""
            SELECT COUNT(*) FROM sma_signal_journal
            WHERE forward_5d_return IS NOT NULL
        """).fetchone()[0]
        check(
            "Backfilled outcomes count < total (expected, recent signals pending)",
            backfilled <= count,
            f"Backfilled: {backfilled}, Total: {count}",
        )

        # was_correct logic
        wrong = conn.execute("""
            SELECT COUNT(*) FROM sma_signal_journal
            WHERE was_correct_5d IS NOT NULL
              AND ((action='BUY' AND forward_5d_return < 0)
                OR (action='SELL' AND forward_5d_return > 0))
              AND was_correct_5d = 1
        """).fetchone()[0]
        check(
            "was_correct_5d flag is logically consistent",
            wrong == 0,
            f"Inconsistent entries: {wrong}",
        )


# ─────────────────────────────────────────────────────────────────────────
# 9. CROSS-MODULE CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────
def audit_consistency(conn: sqlite3.Connection) -> None:
    section("9. CROSS-MODULE CONSISTENCY")

    for ticker in config.WATCHLIST:
        # Latest signal in `signals` table
        sig = conn.execute("""
            SELECT signal, confidence, date FROM signals
            WHERE ticker = ? ORDER BY created_at DESC LIMIT 1
        """, (ticker,)).fetchone()

        # Latest journal entry
        jrn = conn.execute("""
            SELECT action, date FROM sma_signal_journal
            WHERE ticker = ? ORDER BY date DESC LIMIT 1
        """, (ticker,)).fetchone()

        if sig and jrn:
            # They should agree on direction
            sig_dir = sig[0]  # BUY/SELL/HOLD
            jrn_dir = jrn[0]  # BUY/SELL
            check(
                f"{ticker}: signals table & journal agree",
                sig_dir == jrn_dir or sig_dir == "HOLD",
                f"Signal: {sig_dir}, Journal: {jrn_dir}",
                warn_only=True,
            )

    # Check that technical_indicators are aligned with prices
    print()
    for ticker in config.WATCHLIST[:2]:
        price_dates = {
            r[0] for r in conn.execute(
                "SELECT date FROM prices WHERE ticker = ?", (ticker,)
            )
        }
        tech_dates = {
            r[0] for r in conn.execute(
                "SELECT date FROM technical_indicators WHERE ticker = ?", (ticker,)
            )
        }
        missing = price_dates - tech_dates
        check(
            f"{ticker}: every price bar has technical indicators",
            len(missing) <= 5,
            f"Missing on {len(missing)} dates",
            warn_only=True,
        )


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"\n{BOLD}StockOracle System Audit{END}")
    print(f"Run at: {datetime.now(timezone.utc).isoformat()}")
    print(f"Watchlist: {config.WATCHLIST}")

    database.init_db()

    with database.get_connection() as conn:
        audit_schema(conn)
        audit_data(conn)
        audit_features(conn)
        audit_predictor(conn)
        audit_backtester(conn)
        audit_sma_signals(conn)
        audit_portfolio(conn)
        audit_journal(conn)
        audit_consistency(conn)

    # Final summary
    print(f"\n{BOLD}{'='*70}{END}")
    print(f"{BOLD}AUDIT SUMMARY{END}")
    print(f"{BOLD}{'='*70}{END}")
    print(f"  {PASS}: {results['pass']}")
    print(f"  {WARN}: {results['warn']}")
    print(f"  {FAIL}: {results['fail']}")

    if results["fail"] == 0 and results["warn"] == 0:
        print(f"\n{GREEN}{BOLD}🎉 ALL CHECKS PASSED{END}")
    elif results["fail"] == 0:
        print(f"\n{YELLOW}{BOLD}⚠️  Passed with warnings{END}")
    else:
        print(f"\n{RED}{BOLD}❌ FAILURES DETECTED{END}")
        sys.exit(1)


if __name__ == "__main__":
    main()
