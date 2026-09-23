"""Journal and forward-performance tracking for SMA-only signals."""

from __future__ import annotations

from datetime import datetime, timezone

import config

log = config.get_logger(__name__)


def log_signal(
    conn,
    ticker: str,
    regime: str,
    sma50: float | None,
    sma200: float | None,
    close_price: float | None,
    action: str,
) -> None:
    """Insert or replace today's SMA signal for a ticker."""
    signal_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn.execute(
        """
        INSERT OR REPLACE INTO sma_signal_journal
            (date, ticker, regime, sma50, sma200, close_at_signal, action)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (signal_date, ticker, regime, sma50, sma200, close_price, action),
    )


def _future_close(conn, ticker: str, signal_date: str, days_forward: int) -> float | None:
    """Return the first daily close on or after the calendar target date."""
    row = conn.execute(
        """
        SELECT close
        FROM prices
        WHERE ticker = ? AND date >= date(?, ?)
        ORDER BY date ASC
        LIMIT 1
        """,
        (ticker, signal_date, f"+{days_forward} days"),
    ).fetchone()
    return float(row["close"]) if row is not None and row["close"] is not None else None


def backfill_forward_returns(conn) -> None:
    """Fill available 5-, 20-, and 60-day outcomes for journal rows."""
    rows = conn.execute(
        """
        SELECT id, date, ticker, action, close_at_signal,
               forward_5d_return, forward_20d_return, forward_60d_return
        FROM sma_signal_journal
        ORDER BY date, ticker
        """
    ).fetchall()

    for row in rows:
        if row["close_at_signal"] is None:
            continue

        updates: dict[str, float | int] = {}
        for days_forward, column in (
            (5, "forward_5d_return"),
            (20, "forward_20d_return"),
            (60, "forward_60d_return"),
        ):
            if row[column] is not None:
                continue
            future_close = _future_close(conn, row["ticker"], row["date"], days_forward)
            if future_close is None:
                continue
            forward_return = (future_close / float(row["close_at_signal"]) - 1.0) * 100.0
            updates[column] = forward_return

        if "forward_5d_return" in updates:
            forward_5d = float(updates["forward_5d_return"])
            updates["was_correct_5d"] = int(
                (row["action"] == "BUY" and forward_5d > 0)
                or (row["action"] == "SELL" and forward_5d < 0)
            )

        if updates:
            assignments = ", ".join(f"{column} = ?" for column in updates)
            conn.execute(
                f"UPDATE sma_signal_journal SET {assignments} WHERE id = ?",
                (*updates.values(), row["id"]),
            )


def print_journal_summary(conn) -> None:
    """Print signal counts, hit rate, average returns, and recent outcomes."""
    total_signals = conn.execute("SELECT COUNT(*) AS n FROM sma_signal_journal").fetchone()["n"]
    signals_with_5d = conn.execute(
        "SELECT COUNT(*) AS n FROM sma_signal_journal WHERE forward_5d_return IS NOT NULL"
    ).fetchone()["n"]
    signals_with_20d = conn.execute(
        "SELECT COUNT(*) AS n FROM sma_signal_journal WHERE forward_20d_return IS NOT NULL"
    ).fetchone()["n"]
    signals_with_60d = conn.execute(
        "SELECT COUNT(*) AS n FROM sma_signal_journal WHERE forward_60d_return IS NOT NULL"
    ).fetchone()["n"]
    buy_outcomes = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM sma_signal_journal
        WHERE action = 'BUY' AND forward_5d_return IS NOT NULL
        """
    ).fetchone()["n"]
    buy_wins = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM sma_signal_journal
        WHERE action = 'BUY' AND forward_5d_return IS NOT NULL AND forward_5d_return > 0
        """
    ).fetchone()["n"]
    hit_rate = buy_wins / buy_outcomes * 100 if buy_outcomes else 0.0

    regime_averages = conn.execute(
        """
        SELECT regime, AVG(forward_5d_return) AS average_return
        FROM sma_signal_journal
        WHERE forward_5d_return IS NOT NULL
        GROUP BY regime
        """
    ).fetchall()
    averages = {row["regime"]: row["average_return"] for row in regime_averages}

    print("=== SMA Signal Journal Summary ===")
    print(f"Total signals logged:        {total_signals}")
    print(f"Signals with 5d outcome:     {signals_with_5d}")
    print(f"Signals with 20d outcome:    {signals_with_20d}")
    print(f"Signals with 60d outcome:    {signals_with_60d}")
    print(f"Hit rate:                    {hit_rate:.1f}%")
    print(f"Average 5d return bullish:   {averages.get('bullish', 0.0):+.2f}%")
    print(f"Average 5d return bearish:   {averages.get('bearish', 0.0):+.2f}%")
    print()
    print("Recent signals:")
    print("Date        Ticker  Regime    Action  Close       5d Ret")
    recent = conn.execute(
        """
        SELECT date, ticker, regime, action, close_at_signal, forward_5d_return
        FROM sma_signal_journal
        ORDER BY date DESC, id DESC
        LIMIT 10
        """
    ).fetchall()
    for row in recent:
        close = f"{row['close_at_signal']:.2f}" if row["close_at_signal"] is not None else "N/A"
        forward = (
            f"{row['forward_5d_return']:+.2f}%"
            if row["forward_5d_return"] is not None
            else "(pending)"
        )
        print(f"{row['date']:<11} {row['ticker']:<7} {row['regime']:<9} {row['action']:<7} {close:<11} {forward}")

    log.info(
        "SMA journal: %d signals, %d with 5d outcome, %.1f%% BUY hit rate",
        total_signals,
        signals_with_5d,
        hit_rate,
    )