"""Portfolio risk controls used by the allocation pipeline and dashboard."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime

import config

log = config.get_logger(__name__)


def get_portfolio_value(conn: sqlite3.Connection, current_prices: dict[str, float]) -> float:
    """Return tracked positions at current prices plus the latest recorded cash."""
    rows = conn.execute("SELECT ticker, shares FROM open_positions").fetchall()
    positions_value = sum(
        float(row["shares"]) * float(current_prices[row["ticker"]])
        for row in rows
        if row["ticker"] in current_prices
    )
    cash_row = conn.execute(
        "SELECT cash FROM portfolio_history ORDER BY date DESC, id DESC LIMIT 1"
    ).fetchone()
    cash = float(cash_row["cash"]) if cash_row is not None else 0.0
    return positions_value + cash


def get_portfolio_peak(conn: sqlite3.Connection) -> float:
    """Return the highest historical portfolio value, including tracked peaks."""
    row = conn.execute("SELECT MAX(peak_value) AS peak FROM portfolio_history").fetchone()
    return float(row["peak"]) if row is not None and row["peak"] is not None else 0.0


def check_portfolio_stop_loss(
    conn: sqlite3.Connection, current_prices: dict[str, float]
) -> bool:
    """Return whether the portfolio is down at least the configured peak loss."""
    current_value = get_portfolio_value(conn, current_prices)
    peak_value = get_portfolio_peak(conn)
    triggered = peak_value > 0 and current_value <= peak_value * (1.0 - config.PORTFOLIO_STOP_LOSS_PCT)
    log.info(
        "Risk controls: portfolio stop-loss %s (current value $%.2f vs peak $%.2f)",
        "TRIGGERED" if triggered else "NOT triggered",
        current_value,
        peak_value,
    )
    return triggered


def check_position_stop_loss(entry_price: float, current_price: float) -> bool:
    """Return whether a position has fallen at least the configured loss."""
    return current_price <= entry_price * (1.0 - config.POSITION_STOP_LOSS_PCT)


def check_vix_regime(conn: sqlite3.Connection) -> tuple[float | None, float]:
    """Return the latest VIX and its exposure multiplier, degrading to full exposure."""
    row = conn.execute(
        "SELECT vix FROM macro_indicators WHERE vix IS NOT NULL ORDER BY date DESC LIMIT 1"
    ).fetchone()
    if row is None:
        log.info("Risk controls: VIX unavailable; exposure multiplier = 1.0")
        return None, 1.0
    vix = float(row["vix"])
    if vix > config.VIX_EXTREME:
        multiplier = 0.0
    elif vix >= config.VIX_HIGH:
        multiplier = 0.5
    else:
        multiplier = 1.0
    log.info("Risk controls: VIX=%.1f -> exposure multiplier = %.1f", vix, multiplier)
    return vix, multiplier


def check_time_based_exit(entry_date: str, current_price: float, entry_price: float) -> bool:
    """Return whether an old position has failed to reach its target return."""
    try:
        held_days = (date.today() - datetime.fromisoformat(entry_date).date()).days
    except ValueError:
        held_days = (date.today() - date.fromisoformat(entry_date[:10])).days
    return held_days > config.MAX_HOLDING_DAYS and current_price < entry_price * (1.0 + config.MIN_HOLDING_RETURN)


def apply_risk_controls(
    conn: sqlite3.Connection,
    target_allocation: dict[str, float],
    current_prices: dict[str, float],
) -> dict[str, float]:
    """Apply portfolio, position, time, and VIX controls to an allocation."""
    if check_portfolio_stop_loss(conn, current_prices):
        log.info("Risk controls: portfolio stop-loss triggered; exiting all positions to cash")
        log.info(
            "Risk controls: final allocation has 0 positions (was %d)",
            len(target_allocation),
        )
        return {}

    _, exposure_multiplier = check_vix_regime(conn)
    allocation = dict(target_allocation)
    position_exits: list[str] = []
    time_exits: list[str] = []
    rows = conn.execute(
        "SELECT ticker, entry_date, entry_price FROM open_positions"
    ).fetchall()
    for row in rows:
        ticker = row["ticker"]
        current_price = current_prices.get(ticker)
        if current_price is None:
            continue
        position_exit = check_position_stop_loss(float(row["entry_price"]), float(current_price))
        time_exit = check_time_based_exit(
            row["entry_date"], float(current_price), float(row["entry_price"])
        )
        if position_exit or time_exit:
            allocation.pop(ticker, None)
            detail = f"{ticker} {(float(current_price) / float(row['entry_price']) - 1.0):+.0%}"
            if position_exit:
                position_exits.append(detail)
            if time_exit:
                time_exits.append(detail)

    if position_exits:
        log.info(
            "Risk controls: %d positions exited via position stop-loss (%s)",
            len(position_exits), ", ".join(position_exits),
        )
    if time_exits:
        log.info(
            "Risk controls: %d positions exited via time-based exit (%s)",
            len(time_exits), ", ".join(time_exits),
        )
    if exposure_multiplier != 1.0:
        allocation = {
            ticker: weight * exposure_multiplier for ticker, weight in allocation.items()
        }
    log.info(
        "Risk controls: final allocation has %d positions (was %d)",
        len([weight for weight in allocation.values() if weight > 0]),
        len(target_allocation),
    )
    return {ticker: weight for ticker, weight in allocation.items() if weight > 0}


def sync_open_positions(
    conn: sqlite3.Connection,
    allocation: dict[str, float],
    current_prices: dict[str, float],
    as_of: str | None = None,
) -> list[str]:
    """Record allocation holdings while preserving entry data for existing positions."""
    checked_at = as_of or date.today().isoformat()
    existing = {
        row["ticker"]: row
        for row in conn.execute("SELECT * FROM open_positions").fetchall()
    }
    for ticker, weight in allocation.items():
        price = current_prices.get(ticker)
        if price is None:
            continue
        row = existing.get(ticker)
        if row is None:
            conn.execute(
                "INSERT INTO open_positions (ticker, entry_date, entry_price, shares, last_checked) VALUES (?, ?, ?, ?, ?)",
                (ticker, checked_at, float(price), float(weight), checked_at),
            )
        else:
            conn.execute(
                "UPDATE open_positions SET shares = ?, last_checked = ? WHERE ticker = ?",
                (float(weight), checked_at, ticker),
            )
    exits = [ticker for ticker in existing if ticker not in allocation]
    if exits:
        conn.executemany("DELETE FROM open_positions WHERE ticker = ?", [(ticker,) for ticker in exits])
    return exits