"""Portfolio allocation for SMA-only signals."""

from __future__ import annotations

import config

log = config.get_logger(__name__)


def _sector_for_ticker(conn, ticker: str) -> str:
    """Return the most recent known sector for a ticker."""
    row = conn.execute(
        """
        SELECT sector
        FROM fundamentals
        WHERE ticker = ? AND sector IS NOT NULL AND TRIM(sector) != ''
        ORDER BY date DESC
        LIMIT 1
        """,
        (ticker,),
    ).fetchone()
    return str(row[0]).strip() if row else "Unknown"


def compute_portfolio_allocation(conn, signals: dict) -> dict:
    """Return equal-weight allocations for the strongest sector-limited signals."""
    bullish = {
        ticker: signal
        for ticker, signal in signals.items()
        if signal.get("regime") == "bullish"
    }
    for signal in bullish.values():
        sma50, sma200 = signal.get("sma50"), signal.get("sma200")
        signal["strength"] = (sma50 - sma200) / sma200 if sma50 and sma200 else 0

    sorted_tickers = sorted(
        bullish,
        key=lambda ticker: bullish[ticker]["strength"],
        reverse=True,
    )
    sectors = {ticker: _sector_for_ticker(conn, ticker) for ticker in sorted_tickers}
    selected = []
    sector_counts: dict[str, int] = {}
    for ticker in sorted_tickers:
        sector = sectors[ticker]
        if sector_counts.get(sector, 0) >= config.MAX_POSITIONS_PER_SECTOR:
            continue
        selected.append(ticker)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) == config.MAX_POSITIONS:
            break

    if not selected:
        log.info("Portfolio allocation: no bullish positions; total exposure=0%%, cash=100%%")
        return {}

    # Equal weights can exceed a sector cap when fewer than MAX_POSITIONS are
    # selected, so remove the weakest name from any violating sector and retry.
    while selected:
        per_position = min(config.MAX_POSITION_PCT, 1.0 / len(selected))
        sector_weights: dict[str, float] = {}
        for ticker in selected:
            sector = sectors[ticker]
            sector_weights[sector] = sector_weights.get(sector, 0.0) + per_position
        violating_sectors = {
            sector
            for sector, weight in sector_weights.items()
            if weight > config.MAX_SECTOR_EXPOSURE
        }
        if not violating_sectors:
            break
        weakest = next(
            ticker
            for ticker in reversed(selected)
            if sectors[ticker] in violating_sectors
        )
        selected.remove(weakest)

    if not selected:
        return {}

    per_position = min(config.MAX_POSITION_PCT, 1.0 / len(selected))
    allocation = {ticker: per_position for ticker in selected}
    sector_counts = {}
    for ticker in selected:
        sector = sectors[ticker]
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    sector_summary = ", ".join(
        f"{count} {sector.lower()} positions" for sector, count in sorted(sector_counts.items())
    )
    log.info("Sector limits applied: %s", sector_summary)

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
