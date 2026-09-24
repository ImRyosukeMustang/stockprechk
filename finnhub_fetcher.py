"""Finnhub data fetchers for news, earnings, financials, and company profiles.

Setup:
    pip install finnhub-python

Store credentials locally in the gitignored ``finnhub_keys.py`` file:
    FINNHUB_API_KEY = "your-api-key"

The ``FINNHUB_API_KEY`` environment variable is also supported. Finnhub's
free tier allows 60 requests per minute, so every request is spaced by 1.1
seconds and a rate-limit response gets one retry after a 60-second pause.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from typing import Any

import config
import database

log = config.get_logger(__name__)

try:
    from finnhub_keys import FINNHUB_API_KEY
except Exception:
    FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

if not FINNHUB_API_KEY:
    log.warning("No Finnhub API key found — Finnhub fetcher will be skipped.")

_client: Any = None
_last_call_at = 0.0
_MIN_SECONDS_BETWEEN_CALLS = 1.1
_RATE_LIMIT_SLEEP_SECONDS = 60


def _get_client() -> Any:
    global _client
    if _client is not None:
        return _client
    if not config.FINNHUB_ENABLED or not FINNHUB_API_KEY:
        return None
    try:
        import finnhub
    except ImportError:
        log.warning("finnhub-python is not installed — Finnhub fetcher will be skipped.")
        return None
    _client = finnhub.Client(api_key=FINNHUB_API_KEY)
    return _client


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    return status_code == 429 or "429" in str(exc) or "rate limit" in str(exc).lower()


def _call_finnhub(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call Finnhub with free-tier spacing and one 429 retry."""
    global _last_call_at
    wait = _MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    try:
        result = method(*args, **kwargs)
        _last_call_at = time.monotonic()
        return result
    except Exception as exc:
        if not _is_rate_limit_error(exc):
            raise
        log.warning("Finnhub rate limit reached; retrying in 60 seconds.")
        time.sleep(_RATE_LIMIT_SLEEP_SECONDS)
        result = method(*args, **kwargs)
        _last_call_at = time.monotonic()
        return result


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def fetch_company_news(conn: Any, ticker: str, days: int = 7) -> int:
    """Fetch company news from Finnhub for the last N days."""
    client = _get_client()
    if client is None:
        return 0

    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    articles = _call_finnhub(
        client.company_news,
        ticker,
        _from=start_date.isoformat(),
        to=end_date.isoformat(),
    ) or []
    count = 0
    for article in articles:
        url = article.get("url") or f"https://finnhub.io/{article.get('id', '')}"
        published_at = article.get("datetime")
        if isinstance(published_at, (int, float)):
            published_at = datetime.fromtimestamp(published_at, tz=timezone.utc).isoformat()
        if database.insert_news_item(
            conn,
            ticker=ticker,
            source="finnhub",
            title=article.get("headline") or "Untitled Finnhub article",
            url=url,
            published_at=str(published_at) if published_at else None,
            summary=article.get("summary"),
        ):
            count += 1
    log.info("Finnhub news: %d new items for %s", count, ticker)
    return count


def fetch_earnings_calendar(conn: Any, days_forward: int = 30, days_back: int = 7) -> int:
    """Fetch watchlist earnings for the requested date range."""
    client = _get_client()
    if client is None:
        return 0

    start_date = date.today() - timedelta(days=days_back)
    end_date = date.today() + timedelta(days=days_forward)
    response = _call_finnhub(
        client.earnings_calendar,
        _from=start_date.isoformat(),
        to=end_date.isoformat(),
        symbol="",
        international=False,
    ) or {}
    watchlist = set(config.WATCHLIST)
    count = 0
    for entry in response.get("earningsCalendar", []):
        ticker = entry.get("symbol")
        earnings_date = entry.get("date")
        if not ticker or not earnings_date or ticker not in watchlist:
            continue
        if database.insert_earnings_calendar_entry(
            conn,
            ticker=ticker,
            earnings_date=earnings_date,
            eps_estimate=_number(entry.get("epsEstimate")),
            eps_actual=_number(entry.get("epsActual")),
            revenue_estimate=_number(entry.get("revenueEstimate")),
            revenue_actual=_number(entry.get("revenueActual")),
            hour=entry.get("hour"),
        ):
            count += 1
    log.info("Finnhub earnings calendar: %d new entries", count)
    return count


def fetch_basic_financials(conn: Any, ticker: str) -> dict:
    """Fetch basic financials and store today's snapshot when absent."""
    client = _get_client()
    if client is None:
        return {}

    response = _call_finnhub(client.company_basic_financials, ticker, "all") or {}
    metrics = response.get("metric", {})
    result = {
        "ticker": ticker,
        "pe_ratio": _number(metrics.get("peBasicExclExtraTTM")),
        "eps": _number(metrics.get("epsBasicExclExtraItemsTTM")),
        "market_cap": _number(metrics.get("marketCapitalization")),
        "beta": _number(metrics.get("beta")),
        "52_week_high": _number(metrics.get("52WeekHigh")),
        "52_week_low": _number(metrics.get("52WeekLow")),
    }
    today = datetime.now(timezone.utc).date().isoformat()
    existing = conn.execute(
        "SELECT 1 FROM fundamentals WHERE ticker = ? AND date = ? LIMIT 1",
        (ticker, today),
    ).fetchone()
    if existing is None:
        database.insert_fundamentals(
            conn,
            ticker=ticker,
            date=today,
            market_cap=result["market_cap"],
            pe_ratio=result["pe_ratio"],
            eps=result["eps"],
            dividend_yield=None,
            sector=None,
            industry=None,
            raw_json=json.dumps({"source": "finnhub", "metrics": metrics}, default=str),
            beta=result["beta"],
        )
    return result


def fetch_company_profile(conn: Any, ticker: str) -> dict:
    """Fetch and return Finnhub's company profile for a ticker."""
    del conn
    client = _get_client()
    if client is None:
        return {}

    profile = _call_finnhub(client.company_profile2, symbol=ticker) or {}
    return {
        "ticker": ticker,
        "name": profile.get("name"),
        "industry": profile.get("finnhubIndustry"),
        "exchange": profile.get("exchange"),
        "logo": profile.get("logo"),
        "market_cap": _number(profile.get("marketCapitalization")),
    }