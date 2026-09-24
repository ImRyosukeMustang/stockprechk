"""
main.py — StockOracle orchestrator.

Phase 1+2+3 scope: fetch raw data into SQLite, compute derived signals
(sentiment, technical indicators), then train/refresh the ML predictor and
run the decision engine to produce a BUY/SELL/HOLD call. Backtesting
(backtester.py) is deliberately NOT part of this live pipeline — it's a
separate, heavier, offline validation step you run on demand
(`python backtester.py`) before trusting any signal this produces.
llm_analyst.py (written thesis), scheduler.py, and dashboard.py are Phase 4.

Usage:
    python main.py                 # run once for the full watchlist
    python main.py AAPL MSFT       # run once for specific tickers only
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from datetime import date, datetime, timedelta

import config
import data_fetcher
import database
import decision_engine
import journal
import macro_fetcher
import predictor
import portfolio
import pattern_analysis
import pattern_detector
import risk_controls
import sentiment
import technical

log = config.get_logger(__name__)

# Optional: ticker -> company name, used to improve news-search relevance.
# Not exhaustive — data_fetcher falls back to the bare ticker if a name
# isn't listed here.
COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
}


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    """Yield successive ticker chunks of at most ``size`` items."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _latest_prices(conn, tickers: set[str]) -> dict[str, float]:
    """Load the latest stored daily close for each ticker."""
    prices: dict[str, float] = {}
    for ticker in tickers:
        row = conn.execute(
            "SELECT close FROM prices WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        if row is not None and row["close"] is not None:
            prices[ticker] = float(row["close"])
    return prices


def run_pipeline(tickers: list[str] | None = None) -> dict[str, dict[str, int]]:
    """
    Run one full Phase-1 pipeline pass: fetch prices/fundamentals/news/reddit
    for each ticker and store them in SQLite. Returns a per-ticker summary of
    how many items were fetched from each source, for logging/inspection.

    Safe to call repeatedly — all inserts are dedup-safe at the DB layer.
    """
    tickers = tickers or config.WATCHLIST
    database.init_db()

    summary: dict[str, dict[str, int]] = {}
    finnhub_tickers = set(tickers[: config.FINNHUB_MAX_TICKERS_PER_RUN])
    successful_count = 0

    with database.get_connection() as conn:
        run_id = database.start_pipeline_run(conn)
        log.info(
            "Starting pipeline run #%d for %d ticker(s): %s (DRY_RUN=%s)",
            run_id,
            len(tickers),
            ", ".join(tickers),
            config.DRY_RUN,
        )

        try:
            macro_fetcher.fetch_all_macro(conn, date.today() - timedelta(days=365 * 5))
            for chunk_number, chunk in enumerate(_chunks(tickers, 40), start=1):
                for index, ticker in enumerate(chunk, start=(chunk_number - 1) * 40 + 1):
                    try:
                        company_name = COMPANY_NAMES.get(ticker.upper())
                        log.info("--- %s (%d/%d) ---", ticker, index, len(tickers))
                        fetch_counts = data_fetcher.fetch_all_for_ticker(conn, ticker, company_name)
                        if config.FINNHUB_ENABLED and ticker in finnhub_tickers:
                            fetch_counts.update(data_fetcher.fetch_all_finnhub_for_ticker(conn, ticker))

                        # Phase 2: turn what we just fetched (plus prior history) into signals.
                        sentiment_results = sentiment.analyze_ticker_sentiment(conn, ticker)
                        sentiment.categorize_news_events(conn, ticker)
                        technical_count = technical.compute_indicators_for_ticker(conn, ticker)
                        pattern_detector.detect_and_store(conn, ticker)

                        # Phase 3: retrain/refresh the predictor and combine everything
                        # into a BUY/SELL/HOLD call via decision_engine.
                        train_summary = predictor.train_model(conn, ticker)
                        probability_up = predictor.predict_latest(conn, ticker) if train_summary else None
                        decision = decision_engine.generate_signal(conn, ticker)
                        pattern = pattern_analysis.compute_probability(conn, ticker)
                        if decision.get("signal_id"):
                            enriched_reasoning = (
                                f"{decision['reasoning']} | Pattern memory: {pattern['reasoning']} "
                                f"Probability up={pattern['probability_up']:.0%}, confidence={pattern['confidence']:.0%}."
                            )
                            database.update_signal_reasoning(conn, decision["signal_id"], enriched_reasoning)

                        summary[ticker] = {
                            **fetch_counts,
                            "sentiment_news": int(sentiment_results["news"]),
                            "sentiment_reddit": int(sentiment_results["reddit"]),
                            "sentiment_llm": int(sentiment_results["llm"]),
                            "technical_dates": technical_count,
                            "trained": bool(train_summary),
                            "probability_up": probability_up,
                            "signal": decision["signal"],
                            "confidence": decision["confidence"],
                            "pattern_probability_up": pattern["probability_up"],
                            "pattern_confidence": pattern["confidence"],
                        }
                        successful_count += 1
                    except Exception as exc:
                        log.error("Ticker %s failed; continuing: %s", ticker, exc)
                conn.commit()
                processed = min(chunk_number * 40, len(tickers))
                log.info("[%d/%d] Chunk %d complete", processed, len(tickers), chunk_number)
                if processed < len(tickers):
                    time.sleep(5)
            if config.FINNHUB_ENABLED:
                data_fetcher.fetch_finnhub_earnings_calendar(conn)
        except Exception as exc:
            database.finish_pipeline_run(conn, run_id, status="failed", detail=str(exc))
            log.error("Pipeline run #%d failed: %s", run_id, exc)
            raise
        else:
            database.finish_pipeline_run(conn, run_id, status="success", detail=None)
            log.info("Pipeline run #%d finished successfully.", run_id)

    log.info("Pipeline complete: %d/%d tickers succeeded", successful_count, len(tickers))
    _log_summary(summary)
    return summary


def _log_summary(summary: dict[str, dict[str, int]]) -> None:
    log.info("=== Pipeline summary ===")
    for ticker, counts in summary.items():
        parts = ", ".join(f"{source}={count}" for source, count in counts.items())
        log.info("%s: %s", ticker, parts)


def run_pipeline_sma_only(
    tickers: list[str] | None = None,
    refresh_fundamentals: bool = False,
) -> dict:
    """Fetch staged daily data and run the SMA-only signal/allocation pipeline."""
    tickers = tickers or config.WATCHLIST
    database.init_db()
    signals: dict[str, dict] = {}
    successful_count = 0
    refresh_weekly = datetime.now().weekday() == 6 or refresh_fundamentals

    with database.get_connection() as conn:
        journal.backfill_forward_returns(conn)
        log.info(
            "Starting SMA-only pipeline for %d tickers (weekly extras=%s, manual refresh=%s)",
            len(tickers), refresh_weekly, refresh_fundamentals,
        )
        for chunk_number, chunk in enumerate(_chunks(tickers, 40), start=1):
            try:
                for index, ticker in enumerate(chunk, start=(chunk_number - 1) * 40 + 1):
                    try:
                        company_name = COMPANY_NAMES.get(ticker.upper())
                        log.info("--- %s (%d/%d) ---", ticker, index, len(tickers))
                        data_fetcher.fetch_prices_only(conn, ticker)
                        technical.compute_indicators_for_ticker(conn, ticker)
                        signals[ticker] = decision_engine.generate_sma_only_signal(conn, ticker)
                        pattern_detector.detect_and_store(conn, ticker)
                        pattern = pattern_analysis.compute_probability(conn, ticker)
                        latest_prices = database.get_price_history(conn, ticker, limit=1)
                        close_price = float(latest_prices[0]["close"]) if latest_prices else None
                        signal = signals[ticker]
                        journal.log_signal(
                            conn,
                            ticker,
                            signal["regime"],
                            signal["sma50"],
                            signal["sma200"],
                            close_price,
                            "BUY" if signal["regime"] == "bullish" else "SELL",
                        )
                        if signal.get("signal_id"):
                            database.update_signal_reasoning(
                                conn,
                                signal["signal_id"],
                                f"{signal.get('reasoning', '')} | Pattern memory: {pattern['reasoning']} "
                                f"Probability up={pattern['probability_up']:.0%}, confidence={pattern['confidence']:.0%}.",
                            )
                        successful_count += 1
                    except Exception as exc:
                        log.error("Ticker %s failed; continuing: %s", ticker, exc)
                conn.commit()
            except Exception as exc:
                log.error("Chunk %d failed; continuing: %s", chunk_number, exc)
            processed = min(chunk_number * 40, len(tickers))
            log.info("[%d/%d] Chunk %d complete", processed, len(tickers), chunk_number)
            if processed < len(tickers):
                time.sleep(5)

        news_tickers = [
            ticker for ticker in sorted(
                signals,
                key=lambda name: signals[name].get("strength", 0.0),
                reverse=True,
            )
            if signals[ticker].get("regime") == "bullish"
        ][: config.NEWS_MAX_TICKERS_PER_RUN]
        log.info("Fetching daily news for %d top-trend ticker(s)", len(news_tickers))
        for ticker in news_tickers:
            company_name = COMPANY_NAMES.get(ticker.upper())
            data_fetcher.fetch_news_for_ticker(conn, ticker, company_name)
            if config.FINNHUB_ENABLED:
                data_fetcher.fetch_finnhub_news_for_ticker(conn, ticker)

        if refresh_weekly:
            log.info("Refreshing weekly fundamentals and company profiles")
            for ticker in tickers:
                company_name = COMPANY_NAMES.get(ticker.upper())
                data_fetcher.fetch_fundamentals(conn, ticker)
                if config.FINNHUB_ENABLED:
                    data_fetcher.fetch_finnhub_weekly_extras(conn, ticker)

        log.info("Pipeline complete: %d/%d tickers succeeded", successful_count, len(tickers))
        if config.FINNHUB_ENABLED:
            data_fetcher.fetch_finnhub_earnings_calendar(conn)

        target_allocation = portfolio.compute_portfolio_allocation(conn, signals)
        tracked_tickers = {
            row["ticker"] for row in conn.execute("SELECT ticker FROM open_positions").fetchall()
        }
        current_prices = _latest_prices(conn, set(target_allocation) | tracked_tickers)
        allocation = risk_controls.apply_risk_controls(
            conn, target_allocation, current_prices
        )
        exits = risk_controls.sync_open_positions(conn, allocation, current_prices)
        vix, _ = risk_controls.check_vix_regime(conn)
        total_value = risk_controls.get_portfolio_value(conn, current_prices)
        peak_value = max(risk_controls.get_portfolio_peak(conn), total_value)
        position_count = len([weight for weight in allocation.values() if weight > 0])
        notes = f"Exits: {', '.join(exits)}" if exits else None
        conn.execute(
            """
            INSERT OR REPLACE INTO portfolio_history
                (date, total_value, cash, peak_value, n_positions, vix_at_close, notes)
            VALUES (date('now'), ?, ?, ?, ?, ?, ?)
            """,
            (total_value, max(0.0, total_value - sum(
                float(row["shares"]) * current_prices[row["ticker"]]
                for row in conn.execute("SELECT ticker, shares FROM open_positions").fetchall()
                if row["ticker"] in current_prices
            )), peak_value, position_count, vix, notes),
        )
        portfolio_summary = portfolio.compute_portfolio_summary(allocation)

    log.info("SMA-only pipeline complete")
    return {
        "signals": signals,
        "allocation": allocation,
        "portfolio_summary": portfolio_summary,
    }


if __name__ == "__main__":
    refresh_fundamentals = "--refresh-fundamentals" in sys.argv[1:]
    sma_only = "--sma-only" in sys.argv[1:] or refresh_fundamentals
    requested_tickers = [
        t.upper() for t in sys.argv[1:]
        if t not in {"--sma-only", "--refresh-fundamentals"}
    ] or None
    if sma_only:
        run_pipeline_sma_only(requested_tickers, refresh_fundamentals=refresh_fundamentals)
    else:
        run_pipeline(requested_tickers)
