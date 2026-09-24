"""Detect categorical technical patterns and record their forward outcomes."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping

import config

log = config.get_logger(__name__)


def _value(row, *names):
    for name in names:
        try:
            value = row[name]
        except (KeyError, IndexError, TypeError):
            continue
        if value is not None:
            try:
                if math.isnan(float(value)):
                    continue
            except (TypeError, ValueError):
                pass
            return value
    return None


def detect_pattern(row) -> dict[str, str]:
    """Classify one row using RSI, MACD, Bollinger, trend, and volume state."""
    rsi = _value(row, "rsi_14")
    macd = _value(row, "macd")
    signal = _value(row, "macd_signal")
    bb_percent = _value(row, "bb_percent")
    close = _value(row, "close")
    sma_200 = _value(row, "sma_200")
    volume = _value(row, "volume")
    average_volume = _value(row, "volume_avg_20d", "volume_sma20", "avg_volume_20d")

    rsi_bucket = "neutral" if rsi is None or 30 <= float(rsi) <= 70 else "oversold" if float(rsi) < 30 else "overbought"
    if macd is None or signal is None or float(macd) == float(signal):
        macd_state = "neutral"
    else:
        macd_state = "bullish_cross" if float(macd) > float(signal) else "bearish_cross"
    if bb_percent is None:
        bb_position = "middle"
    elif float(bb_percent) < 0.33:
        bb_position = "lower"
    elif float(bb_percent) > 0.66:
        bb_position = "upper"
    else:
        bb_position = "middle"
    trend = "above_sma200" if close is not None and sma_200 is not None and float(close) > float(sma_200) else "below_sma200"
    if volume is None or average_volume is None or float(average_volume) <= 0:
        volume_state = "normal"
    elif float(volume) > 1.5 * float(average_volume):
        volume_state = "high"
    elif float(volume) < 0.5 * float(average_volume):
        volume_state = "low"
    else:
        volume_state = "normal"

    return {
        "rsi_bucket": rsi_bucket,
        "macd_state": macd_state,
        "bb_position": bb_position,
        "trend": trend,
        "volume": volume_state,
    }


def pattern_id(pattern: Mapping[str, str]) -> str:
    """Return a deterministic, compact ID for a categorical pattern."""
    fields = ("rsi_bucket", "macd_state", "bb_position", "trend", "volume")
    canonical = json.dumps({field: pattern.get(field) for field in fields}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _description(pattern: Mapping[str, str]) -> str:
    return (
        f"RSI {pattern['rsi_bucket']}, MACD {pattern['macd_state']}, "
        f"Bollinger {pattern['bb_position']}, {pattern['trend'].replace('_', ' ')}, "
        f"{pattern['volume']} volume"
    )


def detect_and_store(conn: sqlite3.Connection, ticker: str, start_date: str | None = None) -> int:
    """Store one pattern and outcome row for each eligible price bar."""
    rows = conn.execute(
        """
        SELECT p.date, p.close, p.volume, p.open,
               t.rsi_14, t.macd, t.macd_signal, t.bb_upper, t.bb_lower, t.sma_200
        FROM prices p
        LEFT JOIN technical_indicators t ON t.ticker = p.ticker AND t.date = p.date
        WHERE p.ticker = ? AND (? IS NULL OR p.date >= ?)
        ORDER BY p.date ASC
        """,
        (ticker, start_date, start_date),
    ).fetchall()
    if not rows:
        return 0

    closes = [float(row["close"]) for row in rows]
    volumes = [float(row["volume"]) for row in rows]
    stored = 0
    for index, row in enumerate(rows):
        bb_width = row["bb_upper"] - row["bb_lower"] if row["bb_upper"] is not None and row["bb_lower"] is not None else None
        enriched = dict(row)
        enriched["bb_percent"] = ((row["close"] - row["bb_lower"]) / bb_width) if bb_width else None
        prior_volumes = volumes[max(0, index - 20):index]
        enriched["volume_avg_20d"] = sum(prior_volumes) / len(prior_volumes) if prior_volumes else None
        detected = detect_pattern(enriched)
        identifier = pattern_id(detected)
        conn.execute(
            """
            INSERT INTO patterns (pattern_id, rsi_bucket, macd_state, bb_position, trend, volume, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pattern_id) DO UPDATE SET description=excluded.description
            """,
            (identifier, detected["rsi_bucket"], detected["macd_state"], detected["bb_position"], detected["trend"], detected["volume"], _description(detected)),
        )
        forward_5 = closes[index + 5] if index + 5 < len(closes) else None
        forward_20 = closes[index + 20] if index + 20 < len(closes) else None
        return_5 = ((forward_5 / closes[index]) - 1) * 100 if forward_5 is not None else None
        return_20 = ((forward_20 / closes[index]) - 1) * 100 if forward_20 is not None else None
        conn.execute(
            """
            INSERT INTO pattern_outcomes
                (pattern_id, ticker, date, forward_5d_return, forward_20d_return, was_up_5d)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(pattern_id, ticker, date) DO UPDATE SET
                forward_5d_return=excluded.forward_5d_return,
                forward_20d_return=excluded.forward_20d_return,
                was_up_5d=excluded.was_up_5d
            """,
            (identifier, ticker, row["date"], return_5, return_20, int(return_5 > 0) if return_5 is not None else None),
        )
        stored += 1
    log.info("Stored %d patterns/outcomes for %s", stored, ticker)
    return stored
