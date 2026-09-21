"""
data_fetcher.py — pulls raw data from external sources and writes it into SQLite.

Sources, each independent of the others (one failing/missing-key doesn't stop
the rest):
  - yfinance      -> price history + fundamentals
  - Google News RSS (via feedparser) -> free, no API key needed
  - NewsAPI       -> optional, requires config.NEWSAPI_KEY
  - Reddit (praw) -> optional, requires config.REDDIT_CLIENT_ID/SECRET

Every public fetch_* function:
  - takes a ticker (and an open sqlite3 connection),
  - is safe to call repeatedly (dedup happens at the DB layer via UNIQUE
    constraints / INSERT OR IGNORE),
  - never raises on a missing optional dependency or missing API key — it
    logs a warning and returns 0 (rows written) instead.

This module intentionally does not decide *when* to fetch (that's main.py's
job) — it just knows *how*.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.parse
from datetime import datetime, timedelta, timezone

import config
import database

log = config.get_logger(__name__)


# ---------------------------------------------------------------------------
# Prices + fundamentals (yfinance)
# ---------------------------------------------------------------------------

def fetch_prices(conn: sqlite3.Connection, ticker: str) -> int:
    """Fetch OHLCV history for `ticker` via yfinance and store new bars.
    Returns the number of bars fetched (not necessarily all new, since
    INSERT OR IGNORE silently skips duplicates)."""
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — skipping price fetch for %s. `pip install yfinance`.", ticker)
        return 0

    try:
        hist = yf.Ticker(ticker).history(
            period=config.PRICE_HISTORY_PERIOD,
            interval=config.PRICE_HISTORY_INTERVAL,
        )
    except Exception as exc:  # network errors, rate limits, bad ticker, etc.
        log.error("Failed to fetch price history for %s: %s", ticker, exc)
        return 0

    if hist.empty:
        log.warning("yfinance returned no price data for %s", ticker)
        return 0

    rows = []
    for idx, row in hist.iterrows():
        rows.append(
            {
                "date": idx.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            }
        )

    database.insert_price_bars(conn, ticker, rows)
    log.info("Fetched %d price bars for %s", len(rows), ticker)
    return len(rows)


def fetch_fundamentals(conn: sqlite3.Connection, ticker: str) -> bool:
    """Fetch a fundamentals snapshot for `ticker` via yfinance. Returns True on success."""
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — skipping fundamentals fetch for %s.", ticker)
        return False

    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:
        log.error("Failed to fetch fundamentals for %s: %s", ticker, exc)
        return False

    if not info:
        log.warning("yfinance returned no fundamentals info for %s", ticker)
        return False

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    database.insert_fundamentals(
        conn,
        ticker=ticker,
        date=today,
        market_cap=info.get("marketCap"),
        pe_ratio=info.get("trailingPE"),
        eps=info.get("trailingEps"),
        dividend_yield=info.get("dividendYield"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        raw_json=json.dumps(info, default=str),
    )
    log.info("Fetched fundamentals snapshot for %s", ticker)
    return True


# ---------------------------------------------------------------------------
# News — Google News RSS (free, no key) + NewsAPI (optional, needs key)
# ---------------------------------------------------------------------------

def fetch_google_news(conn: sqlite3.Connection, ticker: str, company_name: str | None = None) -> int:
    """Fetch recent headlines for `ticker` from Google News RSS via feedparser.
    No API key required. `company_name`, if given, is used alongside the
    ticker to improve search relevance (e.g. 'AAPL' + 'Apple')."""
    try:
        import feedparser
    except ImportError:
        log.warning("feedparser not installed — skipping Google News fetch for %s. `pip install feedparser`.", ticker)
        return 0

    query = f"{ticker} stock"
    if company_name:
        query = f"{company_name} ({ticker}) stock"
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )

    try:
        feed = feedparser.parse(url)
    except Exception as exc:
        log.error("Failed to fetch Google News for %s: %s", ticker, exc)
        return 0

    count = 0
    for entry in feed.entries:
        published_at = None
        if getattr(entry, "published_parsed", None):
            published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
        database.insert_news_item(
            conn,
            ticker=ticker,
            source="google_news",
            title=entry.get("title", ""),
            url=entry.get("link", ""),
            published_at=published_at,
            summary=entry.get("summary"),
        )
        count += 1

    log.info("Fetched %d Google News items for %s", count, ticker)
    return count


def fetch_newsapi(conn: sqlite3.Connection, ticker: str, company_name: str | None = None) -> int:
    """Fetch recent articles for `ticker` from NewsAPI. Requires config.NEWSAPI_KEY.
    Returns 0 (and logs a warning) if the key isn't configured or the request fails."""
    if not config.NEWSAPI_KEY:
        log.warning("NEWSAPI_KEY not set — skipping NewsAPI fetch for %s.", ticker)
        return 0

    try:
        import requests
    except ImportError:
        log.warning("requests not installed — skipping NewsAPI fetch for %s. `pip install requests`.", ticker)
        return 0

    query = company_name or ticker
    since = (datetime.now(timezone.utc) - timedelta(hours=config.NEWS_LOOKBACK_HOURS)).strftime("%Y-%m-%dT%H:%M:%S")
    params = {
        "q": query,
        "from": since,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": config.NEWSAPI_KEY,
    }

    try:
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.error("Failed to fetch NewsAPI results for %s: %s", ticker, exc)
        return 0

    articles = data.get("articles", [])
    count = 0
    for article in articles:
        database.insert_news_item(
            conn,
            ticker=ticker,
            source="newsapi",
            title=article.get("title", ""),
            url=article.get("url", ""),
            published_at=article.get("publishedAt"),
            summary=article.get("description"),
        )
        count += 1

    log.info("Fetched %d NewsAPI items for %s", count, ticker)
    return count


# ---------------------------------------------------------------------------
# Reddit (praw) — optional
# ---------------------------------------------------------------------------

def fetch_reddit_posts(conn: sqlite3.Connection, ticker: str) -> int:
    """Fetch recent posts mentioning `ticker` from the configured subreddits via praw.
    Requires config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET."""
    if not (config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET):
        log.warning("Reddit credentials not set — skipping Reddit fetch for %s.", ticker)
        return 0

    try:
        import praw
    except ImportError:
        log.warning("praw not installed — skipping Reddit fetch for %s. `pip install praw`.", ticker)
        return 0

    try:
        reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT,
        )
    except Exception as exc:
        log.error("Failed to authenticate with Reddit: %s", exc)
        return 0

    count = 0
    for subreddit_name in config.REDDIT_SUBREDDITS:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            for submission in subreddit.search(ticker, limit=config.REDDIT_POST_LIMIT, sort="new"):
                created_utc = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat()
                database.insert_reddit_post(
                    conn,
                    ticker=ticker,
                    subreddit=subreddit_name,
                    post_id=submission.id,
                    title=submission.title,
                    score=submission.score,
                    num_comments=submission.num_comments,
                    created_utc=created_utc,
                    url=f"https://reddit.com{submission.permalink}",
                )
                count += 1
        except Exception as exc:
            log.error("Failed to fetch Reddit posts from r/%s for %s: %s", subreddit_name, ticker, exc)
            continue

    log.info("Fetched %d Reddit posts for %s", count, ticker)
    return count


# ---------------------------------------------------------------------------
# Orchestration for a single ticker (main.py drives calling this per ticker)
# ---------------------------------------------------------------------------

def fetch_all_for_ticker(conn: sqlite3.Connection, ticker: str, company_name: str | None = None) -> dict[str, int]:
    """Run every fetcher for one ticker and return a dict of source -> item count.
    Each source is independently best-effort: one failing doesn't stop the others."""
    results: dict[str, int] = {}
    results["prices"] = fetch_prices(conn, ticker)
    results["fundamentals"] = int(fetch_fundamentals(conn, ticker))
    results["google_news"] = fetch_google_news(conn, ticker, company_name)
    results["newsapi"] = fetch_newsapi(conn, ticker, company_name)
    results["reddit"] = fetch_reddit_posts(conn, ticker)
    return results


if __name__ == "__main__":
    # Quick manual smoke test: fetch everything for the configured watchlist.
    database.init_db()
    with database.get_connection() as conn:
        for tkr in config.WATCHLIST:
            log.info("Fetching all data for %s...", tkr)
            summary = fetch_all_for_ticker(conn, tkr)
            log.info("%s summary: %s", tkr, summary)
