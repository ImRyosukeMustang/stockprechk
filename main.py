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

import config
import data_fetcher
import database
import decision_engine
import predictor
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
            for ticker in tickers:
                company_name = COMPANY_NAMES.get(ticker.upper())
                log.info("--- %s ---", ticker)
                fetch_counts = data_fetcher.fetch_all_for_ticker(conn, ticker, company_name)

                # Phase 2: turn what we just fetched (plus prior history) into signals.
                sentiment_results = sentiment.analyze_ticker_sentiment(conn, ticker)
                technical_count = technical.compute_indicators_for_ticker(conn, ticker)

                # Phase 3: retrain/refresh the predictor and combine everything
                # into a BUY/SELL/HOLD call via decision_engine. Either can
                # legitimately have nothing to do yet (not enough history to
                # train, or no trained model to predict from) — both degrade
                # to None/HOLD rather than raising, same as every earlier step.
                train_summary = predictor.train_model(conn, ticker)
                probability_up = predictor.predict_latest(conn, ticker) if train_summary else None
                decision = decision_engine.generate_signal(conn, ticker)

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
                }
        except Exception as exc:
            database.finish_pipeline_run(conn, run_id, status="failed", detail=str(exc))
            log.error("Pipeline run #%d failed: %s", run_id, exc)
            raise
        else:
            database.finish_pipeline_run(conn, run_id, status="success", detail=None)
            log.info("Pipeline run #%d finished successfully.", run_id)

    _log_summary(summary)
    return summary


def _log_summary(summary: dict[str, dict[str, int]]) -> None:
    log.info("=== Pipeline summary ===")
    for ticker, counts in summary.items():
        parts = ", ".join(f"{source}={count}" for source, count in counts.items())
        log.info("%s: %s", ticker, parts)


if __name__ == "__main__":
    requested_tickers = [t.upper() for t in sys.argv[1:]] or None
    run_pipeline(requested_tickers)
