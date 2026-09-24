"""Query pattern memory and combine it with recent categorized news."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import config
import pattern_detector

log = config.get_logger(__name__)


def get_pattern_stats(conn: sqlite3.Connection, pattern_id: str) -> dict:
    row = conn.execute(
        """
        SELECT COUNT(*) AS count,
               AVG(forward_5d_return) AS avg_5d,
               AVG(was_up_5d) AS win_rate,
               AVG(forward_20d_return) AS avg_20d
        FROM pattern_outcomes
        WHERE pattern_id = ? AND was_up_5d IS NOT NULL
        """,
        (pattern_id,),
    ).fetchone()
    return {
        "count": int(row["count"] or 0),
        "avg_5d": float(row["avg_5d"]) if row["avg_5d"] is not None else 0.0,
        "win_rate": float(row["win_rate"]) if row["win_rate"] is not None else 0.5,
        "avg_20d": float(row["avg_20d"]) if row["avg_20d"] is not None else 0.0,
    }


def _latest_pattern_row(conn: sqlite3.Connection, ticker: str):
    rows = conn.execute(
        """
        SELECT p.date, p.close, p.volume, t.rsi_14, t.macd, t.macd_signal,
               t.bb_upper, t.bb_lower, t.sma_200
        FROM prices p
        LEFT JOIN technical_indicators t ON t.ticker = p.ticker AND t.date = p.date
        WHERE p.ticker = ? ORDER BY p.date DESC LIMIT 21
        """,
        (ticker,),
    ).fetchall()
    if not rows:
        return None
    rows = list(reversed(rows))
    latest = dict(rows[-1])
    volumes = [float(row["volume"]) for row in rows[:-1] if row["volume"] is not None]
    latest["volume_avg_20d"] = sum(volumes) / len(volumes) if volumes else None
    width = latest["bb_upper"] - latest["bb_lower"] if latest["bb_upper"] is not None and latest["bb_lower"] is not None else None
    latest["bb_percent"] = ((latest["close"] - latest["bb_lower"]) / width) if width else None
    return latest


def find_current_pattern(conn: sqlite3.Connection, ticker: str) -> dict | None:
    row = _latest_pattern_row(conn, ticker)
    if row is None:
        return None
    detected = pattern_detector.detect_pattern(row)
    return {
        **detected,
        "pattern_id": pattern_detector.pattern_id(detected),
    }


def get_similar_patterns(conn: sqlite3.Connection, ticker: str, top_k: int = 5) -> list[dict]:
    current = find_current_pattern(conn, ticker)
    if current is None:
        return []
    fields = ("rsi_bucket", "macd_state", "bb_position", "trend", "volume")
    rows = conn.execute("SELECT * FROM patterns WHERE pattern_id != ?", (current["pattern_id"],)).fetchall()
    ranked = []
    for row in rows:
        distance = sum(row[field] != current[field] for field in fields)
        ranked.append({**dict(row), "distance": distance, "stats": get_pattern_stats(conn, row["pattern_id"])})
    ranked.sort(key=lambda item: (item["distance"], -item["stats"]["count"]))
    return ranked[:max(0, top_k)]


def get_active_news_events(conn: sqlite3.Connection, ticker: str, days: int = 7) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT * FROM news_events WHERE ticker = ? AND date >= ? ORDER BY date DESC, id DESC",
        (ticker, cutoff),
    ).fetchall()
    return [dict(row) for row in rows]


def compute_probability(conn: sqlite3.Connection, ticker: str) -> dict:
    current = find_current_pattern(conn, ticker)
    if current is None:
        return {
            "ticker": ticker, "pattern": {}, "pattern_id": "", "pattern_stats": get_pattern_stats(conn, ""),
            "news_events": [], "probability_up": 0.5, "confidence": 0.0,
            "reasoning": "No price history is available for pattern memory.",
        }
    stats = get_pattern_stats(conn, current["pattern_id"])
    events = get_active_news_events(conn, ticker)
    adjustment = max(-0.15, min(0.15, sum(float(event["sentiment"] or 0) * 0.05 for event in events)))
    probability = max(0.1, min(0.9, stats["win_rate"] + adjustment))
    confidence = min(1.0, stats["count"] / 100)
    reasoning = (
        f"Pattern {current['pattern_id']} has {stats['count']} historical samples with "
        f"a {stats['win_rate']:.0%} 5-day hit rate. "
        f"{len(events)} active news event(s) adjust the base probability by {adjustment:+.2f}."
    )
    return {
        "ticker": ticker,
        "pattern": current,
        "pattern_id": current["pattern_id"],
        "pattern_stats": stats,
        "news_events": events,
        "probability_up": probability,
        "confidence": confidence,
        "reasoning": reasoning,
    }
