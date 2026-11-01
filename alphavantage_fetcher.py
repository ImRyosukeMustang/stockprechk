"""Optional Alpha Vantage news and sentiment fetcher."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

import config
import database

log = config.get_logger(__name__)
API_URL = "https://www.alphavantage.co/query"


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _date(value: str | None) -> str:
    if value:
        try:
            return datetime.strptime(value, "%Y%m%dT%H%M%S").date().isoformat()
        except ValueError:
            return value[:10]
    return datetime.now().date().isoformat()


def fetch_news_sentiment(conn: sqlite3.Connection, ticker: str) -> int:
    """Fetch recent news and persist article sentiment for ``ticker``."""
    if not config.ALPHAVANTAGE_API_KEY:
        log.info("ALPHAVANTAGE_API_KEY is not set; skipping news sentiment for %s.", ticker)
        return 0

    try:
        import requests

        response = requests.get(
            API_URL,
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "limit": 50,
                "apikey": config.ALPHAVANTAGE_API_KEY,
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        # Request exceptions may include the URL, which contains the API key.
        log.warning("Alpha Vantage news fetch failed for %s (%s).", ticker, type(exc).__name__)
        return 0

    if not isinstance(payload, dict):
        return 0
    error_message = payload.get("Error Message") or payload.get("Note") or payload.get("Information")
    if error_message:
        log.warning("Alpha Vantage returned an API notice for %s.", ticker)
        return 0

    count = 0
    for article in payload.get("feed", []):
        if not isinstance(article, dict):
            continue
        title = str(article.get("title") or "").strip()
        url = str(article.get("url") or "").strip()
        if not title or not url:
            continue
        article_date = _date(article.get("time_published"))
        ticker_sentiment = next(
            (
                item for item in article.get("ticker_sentiment", [])
                if isinstance(item, dict) and str(item.get("ticker", "")).upper() == ticker.upper()
            ),
            {},
        )
        score = _number(ticker_sentiment.get("ticker_sentiment_score"))
        if score is None:
            score = _number(article.get("overall_sentiment_score"))
        label = ticker_sentiment.get("ticker_sentiment_label") or article.get("overall_sentiment_label")
        inserted = database.insert_news_item(
            conn,
            ticker=ticker,
            source="alphavantage",
            title=title,
            url=url,
            published_at=article.get("time_published"),
            summary=article.get("summary"),
        )
        if inserted and score is not None:
            conn.execute(
                """
                INSERT INTO sentiment_scores (ticker, date, source_type, score, label, detail_json)
                VALUES (?, ?, 'alphavantage', ?, ?, ?)
                """,
                (ticker, article_date, score, label, json.dumps(article, sort_keys=True)),
            )
        count += int(inserted)
    log.info("Stored %d Alpha Vantage news articles for %s.", count, ticker)
    return count
