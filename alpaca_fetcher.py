"""
alpaca_fetcher.py - pulls intraday OHLCV bars and live quotes from Alpaca
and stores them in the `prices_intraday` table.

Uses the free IEX feed (via alpaca-py). Free tier allows 200 req/min.

Functions:
  fetch_bars(conn, ticker, interval, days_back)  - pull bars for one interval
  fetch_latest_quote(conn, ticker)               - pull current bid/ask
  fetch_all_for_ticker(conn, ticker)             - pull hourly + minute bars + quote

Reads API keys from `alpaca_keys.py` (gitignored, local-only).

Safe to call repeatedly - INSERT OR REPLACE on (ticker, interval, timestamp).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Literal

import config
import database

log = config.get_logger(__name__)

# Import keys from the local gitignored file
try:
    from alpaca_keys import ALPACA_API_KEY, ALPACA_SECRET_KEY
except ImportError:
    ALPACA_API_KEY = None
    ALPACA_SECRET_KEY = None
    log.warning("alpaca_keys.py not found - Alpaca fetcher will be skipped.")

# Map our interval strings to Alpaca's timeframe units
_INTERVAL_MAP = {
    "1m": ("Minute", 1),
    "5m": ("Minute", 5),
    "15m": ("Minute", 15),
    "30m": ("Minute", 30),
    "1h": ("Hour", 1),
    "1d": ("Day", 1),
}


def _get_client():
    """Return an initialized Alpaca historical data client, or None if keys missing."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return None
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient
    except ImportError:
        log.warning("alpaca-py not installed - `pip install alpaca-py`.")
        return None
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def fetch_bars(
    conn: sqlite3.Connection,
    ticker: str,
    interval: Literal["1m", "5m", "15m", "30m", "1h", "1d"] = "1h",
    days_back: int = 30,
) -> int:
    """
    Fetch OHLCV bars from Alpaca for one interval and store them.
    Returns number of bars written (0 on failure or missing keys).
    """
    if interval not in _INTERVAL_MAP:
        log.warning("Unsupported interval '%s' for %s - skipping.", interval, ticker)
        return 0

    client = _get_client()
    if client is None:
        return 0

    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed
    except ImportError:
        return 0

    unit_name, amount = _INTERVAL_MAP[interval]
    unit = getattr(TimeFrameUnit, unit_name)

    start = datetime.now(timezone.utc) - timedelta(days=days_back)

    try:
        req = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame(amount=amount, unit=unit),
            start=start,
            feed=DataFeed.IEX,
        )
        bars_df = client.get_stock_bars(req).df
    except Exception as exc:
        log.error("Alpaca bars fetch failed for %s (%s): %s", ticker, interval, exc)
        return 0

    if bars_df.empty:
        log.warning("Alpaca returned no bars for %s (%s).", ticker, interval)
        return 0

    # bars_df has a MultiIndex (symbol, timestamp). Flatten it.
    rows = []
    for (symbol, ts), row in bars_df.iterrows():
        rows.append({
            "ticker": ticker,
            "interval": interval,
            "timestamp": ts.isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
            "vwap": float(row["vwap"]) if "vwap" in row and row["vwap"] == row["vwap"] else None,
            "trade_count": int(row["trade_count"]) if "trade_count" in row else None,
        })

    conn.executemany(
        """
        INSERT OR REPLACE INTO prices_intraday
            (ticker, interval, timestamp, open, high, low, close, volume, vwap, trade_count)
        VALUES (:ticker, :interval, :timestamp, :open, :high, :low, :close, :volume, :vwap, :trade_count)
        """,
        rows,
    )

    log.info("Alpaca: stored %d bars for %s (%s, %dd lookback).", len(rows), ticker, interval, days_back)
    return len(rows)


def fetch_latest_quote(conn: sqlite3.Connection, ticker: str) -> dict | None:
    """Fetch the current bid/ask for a ticker. Returns a dict or None."""
    client = _get_client()
    if client is None:
        return None

    try:
        from alpaca.data.requests import StockLatestQuoteRequest
        from alpaca.data.enums import DataFeed
    except ImportError:
        return None

    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=[ticker], feed=DataFeed.IEX)
        quote_resp = client.get_stock_latest_quote(req)
    except Exception as exc:
        log.error("Alpaca quote fetch failed for %s: %s", ticker, exc)
        return None

    q = quote_resp.get(ticker)
    if q is None:
        return None

    return {
        "ticker": ticker,
        "timestamp": q.timestamp.isoformat() if q.timestamp else None,
        "bid_price": float(q.bid_price) if q.bid_price else None,
        "ask_price": float(q.ask_price) if q.ask_price else None,
        "bid_size": int(q.bid_size) if q.bid_size else None,
        "ask_size": int(q.ask_size) if q.ask_size else None,
    }


def fetch_all_for_ticker(conn: sqlite3.Connection, ticker: str) -> dict[str, int]:
    """
    Convenience: pull hourly bars (30d), 5-minute bars (5d), and the current quote.
    Returns a summary dict of what was fetched.
    """
    return {
        "bars_1h": fetch_bars(conn, ticker, interval="1h", days_back=30),
        "bars_5m": fetch_bars(conn, ticker, interval="5m", days_back=5),
        "quote": 1 if fetch_latest_quote(conn, ticker) else 0,
    }


if __name__ == "__main__":
    database.init_db()
    with database.get_connection() as conn:
        log.info("=== Alpaca smoke test ===")
        for ticker in config.WATCHLIST[:2]:  # just first 2 to keep it fast
            summary = fetch_all_for_ticker(conn, ticker)
            log.info("%s: %s", ticker, summary)

        # Verify what landed in the DB
        counts = conn.execute("""
            SELECT ticker, interval, COUNT(*) as n, MIN(timestamp) as first, MAX(timestamp) as last
            FROM prices_intraday
            GROUP BY ticker, interval
            ORDER BY ticker, interval
        """).fetchall()
        log.info("=== prices_intraday summary ===")
        for row in counts:
            log.info("%s | %s | %d bars | %s -> %s", row[0], row[1], row[2], row[3], row[4])