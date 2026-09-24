"""Fetch and store daily macroeconomic indicators from FRED's public CSV feed."""

from __future__ import annotations

import sqlite3
from datetime import date

import pandas as pd
import config

log = config.get_logger(__name__)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
SERIES = {
    "FEDFUNDS": "fed_funds_rate",
    "DGS10": "ten_year_yield",
    "CPIAUCSL": "cpi",
    "VIXCLS": "vix",
}


def fetch_fred_series(series_id: str, start_date: str | date) -> pd.DataFrame:
    """Return a FRED series as a date-indexed DataFrame, or an empty frame on failure."""
    import requests

    start = start_date.isoformat() if isinstance(start_date, date) else str(start_date)
    try:
        response = requests.get(FRED_URL.format(series_id=series_id), timeout=30)
        response.raise_for_status()
        frame = pd.read_csv(pd.io.common.StringIO(response.text))
    except (OSError, ValueError, requests.RequestException) as exc:
        log.warning("Failed to fetch FRED series %s: %s", series_id, exc)
        return pd.DataFrame(columns=["date", series_id])
    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"] >= pd.Timestamp(start)].copy()
    frame[series_id] = pd.to_numeric(frame[series_id], errors="coerce")
    return frame[["date", series_id]].dropna(subset=["date"])


def fetch_all_macro(conn: sqlite3.Connection, start_date: str | date) -> int:
    """Fetch all configured FRED series and upsert their date-aligned values."""
    merged = None
    for series_id in SERIES:
        frame = fetch_fred_series(series_id, start_date)
        if frame.empty:
            continue
        merged = frame if merged is None else merged.merge(frame, on="date", how="outer")
    if merged is None or merged.empty:
        return 0
    merged = merged.sort_values("date")
    columns = ["fed_funds_rate", "ten_year_yield", "cpi", "vix"]
    merged = merged.rename(columns={series_id: column for series_id, column in SERIES.items()})
    for _, row in merged.iterrows():
        values = [None if pd.isna(row.get(column)) else float(row[column]) for column in columns]
        conn.execute(
            """
            INSERT INTO macro_indicators (date, fed_funds_rate, ten_year_yield, cpi, vix)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                fed_funds_rate=excluded.fed_funds_rate,
                ten_year_yield=excluded.ten_year_yield,
                cpi=excluded.cpi,
                vix=excluded.vix,
                fetched_at=datetime('now')
            """,
            [row["date"].strftime("%Y-%m-%d"), *values],
        )
    log.info("Stored %d macro indicator dates", len(merged))
    return len(merged)


if __name__ == "__main__":
    from datetime import timedelta
    import database

    database.init_db()
    with database.get_connection() as connection:
        fetch_all_macro(connection, date.today() - timedelta(days=365 * 5))