"""Optional Financial Modeling Prep profile and fundamentals fetcher."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import config

log = config.get_logger(__name__)
API_URL = "https://financialmodelingprep.com/stable"
STATEMENT_ENDPOINTS = (
    "income-statement",
    "balance-sheet-statement",
    "cash-flow-statement",
)


def _fetch_json(endpoint: str, ticker: str) -> Any:
    if not config.FMP_API_KEY:
        return None
    try:
        import requests

        response = requests.get(
            f"{API_URL}/{endpoint}",
            params={"symbol": ticker, "limit": 5, "apikey": config.FMP_API_KEY},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        # Request exceptions can include the full URL, including its API key.
        log.warning("FMP %s fetch failed for %s (%s).", endpoint, ticker, type(exc).__name__)
        return None


def fetch_company_profile(conn: sqlite3.Connection, ticker: str) -> int:
    """Fetch and upsert the FMP company profile for one ticker."""
    payload = _fetch_json("profile", ticker)
    if payload is None:
        return 0
    try:
        profile = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(profile, dict) or not profile:
            return 0
        conn.execute(
            """
            INSERT INTO company_profiles (ticker, profile_json, fetched_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(ticker) DO UPDATE SET
                profile_json = excluded.profile_json, fetched_at = excluded.fetched_at
            """,
            (ticker, json.dumps(profile, sort_keys=True)),
        )
        return 1
    except Exception as exc:
        log.warning("Could not store FMP profile for %s (%s).", ticker, type(exc).__name__)
        return 0


def fetch_financial_statements(conn: sqlite3.Connection, ticker: str) -> int:
    """Fetch recent income, balance-sheet, and cash-flow statements."""
    stored = 0
    for statement_type in STATEMENT_ENDPOINTS:
        payload = _fetch_json(statement_type, ticker)
        if not isinstance(payload, list):
            continue
        for statement in payload:
            if not isinstance(statement, dict):
                continue
            report_date = str(statement.get("date") or statement.get("calendarYear") or "unknown")
            period = str(statement.get("period") or "")
            try:
                conn.execute(
                    """
                    INSERT INTO financial_statements
                        (ticker, statement_type, report_date, period, raw_json, fetched_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(ticker, statement_type, report_date, period) DO UPDATE SET
                        raw_json = excluded.raw_json, fetched_at = excluded.fetched_at
                    """,
                    (ticker, statement_type, report_date, period, json.dumps(statement, sort_keys=True)),
                )
                stored += 1
            except Exception as exc:
                log.warning("Could not store FMP %s for %s (%s).", statement_type, ticker, type(exc).__name__)
    return stored


def fetch_key_metrics(conn: sqlite3.Connection, ticker: str) -> int:
    """Fetch and upsert recent FMP key metrics."""
    payload = _fetch_json("key-metrics", ticker)
    if not isinstance(payload, list):
        return 0
    stored = 0
    for metric in payload:
        if not isinstance(metric, dict):
            continue
        report_date = str(metric.get("date") or metric.get("calendarYear") or "unknown")
        period = str(metric.get("period") or "")
        try:
            conn.execute(
                """
                INSERT INTO key_metrics (ticker, report_date, period, raw_json, fetched_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(ticker, report_date, period) DO UPDATE SET
                    raw_json = excluded.raw_json, fetched_at = excluded.fetched_at
                """,
                (ticker, report_date, period, json.dumps(metric, sort_keys=True)),
            )
            stored += 1
        except Exception as exc:
            log.warning("Could not store FMP key metrics for %s (%s).", ticker, type(exc).__name__)
    return stored


def fetch_all_for_ticker(conn: sqlite3.Connection, ticker: str) -> dict[str, int]:
    """Fetch profile, statements, and metrics without failing the pipeline."""
    if not config.FMP_API_KEY:
        log.info("FMP_API_KEY is not set; skipping FMP data for %s.", ticker)
        return {"profile": 0, "statements": 0, "key_metrics": 0}
    return {
        "profile": fetch_company_profile(conn, ticker),
        "statements": fetch_financial_statements(conn, ticker),
        "key_metrics": fetch_key_metrics(conn, ticker),
    }
