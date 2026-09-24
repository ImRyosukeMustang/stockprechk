"""
sentiment.py — sentiment scoring for stored news headlines and Reddit posts.

Two independent, best-effort scorers:
  - FinBERT (HuggingFace `ProsusAI/finbert`, via `transformers`) — fast,
    local, no API cost. Scores each recent news headline and Reddit title
    individually, then aggregates to one score per ticker/date/source.
  - LLM (OpenAI, config.LLM_MODEL_CHEAP) — reads all of today's headlines
    and post titles together and produces one overall sentiment judgement
    with brief reasoning. More expensive per call, so used for a single
    daily aggregate rather than per-item scoring.

Both scorers degrade gracefully: if `transformers`/`torch` aren't
installed, or `config.OPENAI_API_KEY` isn't set, that scorer is skipped
with a warning — the other can still run, and the pipeline never crashes
because a sentiment source is unavailable.

Score convention: every score stored in `sentiment_scores.score` is
normalized to the range -1.0 (very negative) to +1.0 (very positive),
0.0 being neutral/no signal, regardless of which scorer produced it. This
lets decision_engine.py (Phase 3) treat all sentiment rows uniformly.

Known limitation: data_fetcher.py only pulls a recent window of news
(config.NEWS_LOOKBACK_HOURS) and Reddit posts, not a historical archive.
So sentiment_scores will only ever have rows for recent dates — it is not
a source of historical sentiment for backtesting arbitrary past dates.
backtester.py (Phase 3) needs to account for this gap rather than assume
sentiment features exist everywhere prices do.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import config
import database

log = config.get_logger(__name__)

# Lazily-loaded, module-level cache so we only pay FinBERT's model-load cost
# once per process, not once per ticker.
_finbert_pipeline = None
_finbert_load_attempted = False


def _get_finbert_pipeline():
    """Return a cached HuggingFace sentiment-analysis pipeline for FinBERT,
    or None if `transformers`/`torch` aren't installed or loading failed."""
    global _finbert_pipeline, _finbert_load_attempted
    if _finbert_load_attempted:
        return _finbert_pipeline

    _finbert_load_attempted = True
    try:
        from transformers import pipeline
    except ImportError:
        log.warning("transformers not installed — FinBERT sentiment unavailable. `pip install transformers torch`.")
        return None

    try:
        _finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception as exc:
        log.error("Failed to load FinBERT model: %s", exc)
        _finbert_pipeline = None

    return _finbert_pipeline


def _finbert_signed_score(label: str, confidence: float) -> float:
    """Convert FinBERT's (label, confidence) into a single signed score in [-1, 1]."""
    label = label.lower()
    if label == "positive":
        return confidence
    if label == "negative":
        return -confidence
    return 0.0  # neutral carries no directional signal


def score_texts_finbert(texts: list[str]) -> list[tuple[float, str, float]]:
    """
    Score a list of short texts (headlines/titles) with FinBERT.
    Returns a list of (signed_score, label, confidence) tuples, same order
    as input, or an empty list if FinBERT isn't available.
    """
    if not texts:
        return []

    clf = _get_finbert_pipeline()
    if clf is None:
        return []

    try:
        # FinBERT has a token limit; truncate defensively — headlines/titles
        # are short so this rarely matters in practice.
        results = clf(texts, truncation=True, max_length=512)
    except Exception as exc:
        log.error("FinBERT scoring failed: %s", exc)
        return []

    out = []
    for r in results:
        label = r["label"]
        confidence = float(r["score"])
        out.append((_finbert_signed_score(label, confidence), label, confidence))
    return out


def analyze_news_sentiment(conn: sqlite3.Connection, ticker: str, limit: int = 30) -> bool:
    """Score recent stored news headlines for `ticker` with FinBERT and store
    one aggregate row (mean of per-item signed scores) for today's date.
    Returns True if a score was written."""
    items = database.get_recent_news(conn, ticker, limit=limit)
    if not items:
        log.info("No stored news for %s — skipping news sentiment.", ticker)
        return False

    titles = [row["title"] for row in items if row["title"]]
    scored = score_texts_finbert(titles)
    if not scored:
        return False

    avg_score = sum(s for s, _, _ in scored) / len(scored)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    detail = {
        "n_items": len(scored),
        "labels": [label for _, label, _ in scored],
    }
    database.replace_sentiment_score(
        conn, ticker, today, "news", avg_score,
        label=_score_to_label(avg_score), detail_json=json.dumps(detail),
    )
    log.info("News sentiment for %s: %.3f (from %d headlines)", ticker, avg_score, len(scored))
    return True


def analyze_reddit_sentiment(conn: sqlite3.Connection, ticker: str, limit: int = 30) -> bool:
    """Score recent stored Reddit post titles for `ticker` with FinBERT and
    store one aggregate row for today's date. Returns True if written."""
    items = database.get_recent_reddit_posts(conn, ticker, limit=limit)
    if not items:
        log.info("No stored Reddit posts for %s — skipping reddit sentiment.", ticker)
        return False

    titles = [row["title"] for row in items if row["title"]]
    scored = score_texts_finbert(titles)
    if not scored:
        return False

    avg_score = sum(s for s, _, _ in scored) / len(scored)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    detail = {
        "n_items": len(scored),
        "labels": [label for _, label, _ in scored],
    }
    database.replace_sentiment_score(
        conn, ticker, today, "reddit", avg_score,
        label=_score_to_label(avg_score), detail_json=json.dumps(detail),
    )
    log.info("Reddit sentiment for %s: %.3f (from %d posts)", ticker, avg_score, len(scored))
    return True


def analyze_llm_sentiment(conn: sqlite3.Connection, ticker: str, news_limit: int = 15, reddit_limit: int = 15) -> bool:
    """
    Ask an LLM (config.LLM_MODEL_CHEAP) to read today's headlines + Reddit
    titles together and give one overall sentiment score + short reasoning.
    Requires config.OPENAI_API_KEY. Returns True if a score was written.
    """
    if not config.OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY not set — skipping LLM sentiment for %s.", ticker)
        return False

    try:
        from openai import OpenAI
    except ImportError:
        log.warning("openai package not installed — skipping LLM sentiment for %s. `pip install openai`.", ticker)
        return False

    news_items = database.get_recent_news(conn, ticker, limit=news_limit)
    reddit_items = database.get_recent_reddit_posts(conn, ticker, limit=reddit_limit)
    if not news_items and not reddit_items:
        log.info("No stored news or Reddit posts for %s — skipping LLM sentiment.", ticker)
        return False

    headlines = "\n".join(f"- {row['title']}" for row in news_items)
    reddit_titles = "\n".join(f"- {row['title']}" for row in reddit_items)

    prompt = f"""You are a financial sentiment analyst. Read the following recent
headlines and Reddit post titles about {ticker} and judge the OVERALL
market sentiment they convey.

News headlines:
{headlines or "(none)"}

Reddit post titles:
{reddit_titles or "(none)"}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"score": <float from -1.0 (very negative) to 1.0 (very positive)>,
  "label": "<positive|neutral|negative>",
  "reasoning": "<one or two sentence explanation>"}}
"""

    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.LLM_MODEL_CHEAP,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        score = float(parsed["score"])
        label = parsed.get("label", _score_to_label(score))
        reasoning = parsed.get("reasoning", "")
    except Exception as exc:
        log.error("LLM sentiment call failed for %s: %s", ticker, exc)
        return False

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    database.replace_sentiment_score(
        conn, ticker, today, "llm", score, label=label,
        detail_json=json.dumps({"reasoning": reasoning, "model": config.LLM_MODEL_CHEAP}),
    )
    log.info("LLM sentiment for %s: %.3f (%s) — %s", ticker, score, label, reasoning)
    return True


def _score_to_label(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


def analyze_ticker_sentiment(conn: sqlite3.Connection, ticker: str) -> dict[str, bool]:
    """Run all available sentiment scorers for one ticker. Each is independent —
    one being unavailable doesn't block the others. Returns which ones wrote a score."""
    return {
        "news": analyze_news_sentiment(conn, ticker),
        "reddit": analyze_reddit_sentiment(conn, ticker),
        "llm": analyze_llm_sentiment(conn, ticker),
    }


def categorize_news_events(conn: sqlite3.Connection, ticker: str) -> int:
    """Categorize uncategorized headlines from the last seven days with OpenAI."""
    if not config.OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY not set — skipping news event categorization for %s.", ticker)
        return 0
    try:
        from openai import OpenAI
    except ImportError:
        log.warning("openai package not installed — skipping news event categorization.")
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    existing = {
        row["url"]
        for row in conn.execute("SELECT url FROM news_events WHERE ticker = ?", (ticker,)).fetchall()
        if row["url"]
    }
    news_items = database.get_recent_news(conn, ticker, limit=100)
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    categorized = 0
    allowed_types = {"earnings", "analyst_rating", "M&A", "regulatory", "macro_fed", "product_launch", "legal", "geopolitical", "management_change", "other"}
    for item in news_items:
        url = item["url"] or f"news:{item['id']}"
        if url in existing:
            continue
        raw_date = item["published_at"] or item["fetched_at"]
        try:
            item_date = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
            if item_date.tzinfo is None:
                item_date = item_date.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        if item_date < cutoff:
            continue
        prompt = f"""Categorize this headline about {ticker}. Return JSON:
{{
  "event_type": one of ["earnings", "analyst_rating", "M&A", "regulatory", "macro_fed", "product_launch", "legal", "geopolitical", "management_change", "other"],
  "political": 0 or 1,
  "sentiment": -1.0 to 1.0,
  "summary": "one-sentence explanation"
}}

Headline: {item['title']}"""
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL_CHEAP,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            event_type = parsed.get("event_type", "other")
            if event_type not in allowed_types:
                event_type = "other"
            score = max(-1.0, min(1.0, float(parsed.get("sentiment", 0.0))))
            database.insert_news_event(
                conn, ticker, item_date.strftime("%Y-%m-%d"), item["title"], url,
                event_type, score, int(bool(parsed.get("political", 0))), event_type,
                str(parsed.get("summary", "")),
            )
            categorized += 1
        except Exception as exc:
            log.warning("Failed to categorize news for %s: %s", ticker, exc)
    log.info("Categorized %d news events for %s", categorized, ticker)
    return categorized


def analyze_watchlist_sentiment(conn: sqlite3.Connection, tickers: list[str] | None = None) -> dict[str, dict[str, bool]]:
    tickers = tickers or config.WATCHLIST
    results: dict[str, dict[str, bool]] = {}
    for ticker in tickers:
        try:
            results[ticker] = analyze_ticker_sentiment(conn, ticker)
        except Exception as exc:
            log.error("Unexpected error analyzing sentiment for %s: %s", ticker, exc)
            results[ticker] = {"news": False, "reddit": False, "llm": False}
    return results


if __name__ == "__main__":
    database.init_db()
    with database.get_connection() as conn:
        summary = analyze_watchlist_sentiment(conn)
        log.info("=== Sentiment summary ===")
        for ticker, result in summary.items():
            log.info("%s: %s", ticker, result)
