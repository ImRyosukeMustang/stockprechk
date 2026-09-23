"""Portfolio allocation for SMA-only signals."""

from __future__ import annotations

import config

log = config.get_logger(__name__)


def compute_portfolio_allocation(signals: dict) -> dict:
    """Return equal-weight allocations for bullish signals.

    MAX_POSITION_PCT is a per-position cap, not a divisor across the whole
    watchlist. The number of bullish tickers determines the uncapped equal
    weight; MAX_TOTAL_EXPOSURE limits the combined allocation.
    """
    bullish = [
        ticker
        for ticker, signal in signals.items()
        if signal.get("regime") == "bullish"
    ]
    allocation = {ticker: 0.0 for ticker in signals}
    if not bullish:
        log.info("Portfolio allocation: no bullish positions; total exposure=0%%, cash=100%%")
        return allocation

    max_position = max(0.0, float(config.MAX_POSITION_PCT))
    max_exposure = max(0.0, min(1.0, float(getattr(config, "MAX_TOTAL_EXPOSURE", 1.0))))
    n_bullish = len(bullish)
    per_position = min(max_position, max_exposure / max(n_bullish, 1))

    for ticker in bullish:
        allocation[ticker] = per_position

    summary = compute_portfolio_summary(allocation)
    log.info("Portfolio allocation:")
    for ticker, weight in allocation.items():
        if weight > 0:
            log.info("  %s: %.0f%%", ticker, weight * 100)
    log.info(
        "  Total exposure: %.0f%%, Cash: %.0f%%",
        summary["total_exposure"] * 100,
        summary["cash_pct"] * 100,
    )
    return allocation


def compute_portfolio_summary(allocation: dict) -> dict:
    """Return total exposure, cash percentage, and number of positions."""
    total_exposure = sum(max(0.0, float(weight)) for weight in allocation.values())
    total_exposure = min(1.0, total_exposure)
    cash_pct = max(0.0, 1.0 - total_exposure)
    return {
        "total_exposure": total_exposure,
        "cash_pct": cash_pct,
        "n_positions": sum(1 for weight in allocation.values() if float(weight) > 0),
    }
