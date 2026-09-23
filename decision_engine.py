"""
decision_engine.py — combines predictor.py's probability, sentiment.py's
scores, and technical.py's indicators into one BUY/SELL/HOLD signal with a
confidence score and a short structured reasoning string.

Design:
  - Every input is optional. A ticker might have a trained model and no
    fresh sentiment, or sentiment but no model trained yet. Each available
    component contributes a score in [-1, 1] (positive = bullish, negative
    = bearish), and the final composite is a weighted average over only the
    components that are actually present — weights are NOT diluted by
    missing components, so confidence reflects the strength of the evidence
    that exists, not a penalty for data that hasn't arrived yet.
  - If NOTHING is available (no model, no sentiment, no technicals), the
    result is HOLD with confidence 0.0 and reasoning explaining why.
  - `config.MIN_CONFIDENCE_TO_ACT` is a hard floor: even a clearly bullish
    or bearish composite becomes HOLD if confidence doesn't clear it. This
    is the project's stated risk control, not a suggestion.
  - This module NEVER places trades. It writes a row to the `signals` table
    and returns it. What happens with a BUY/SELL signal (paper trade, alert,
    nothing) is entirely up to code outside this module, and `config.DRY_RUN`
    must be respected by any of it.

Weights (tunable in WEIGHTS below):
  - prediction (from predictor.py): 0.5
  - sentiment (mean of news/reddit/llm, from sentiment.py): 0.3
  - technical (from RSI + MACD state): 0.2

Technical sub-score interpretation, spelled out because it's a judgment
call: RSI is treated as a mean-reversion signal (low RSI = oversold = mildly
bullish, high RSI = overbought = mildly bearish) rather than a momentum
signal. MACD is treated as a momentum signal (MACD above its signal line =
bullish state). These two philosophies can and will disagree with each
other — that's intentional; decision_engine averages them rather than
picking one, and the reasoning string reports both so a human reviewing a
signal can see where the disagreement came from.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import config
import database
import predictor

log = config.get_logger(__name__)

WEIGHTS = {
    "prediction": 0.5,
    "sentiment": 0.3,
    "technical": 0.2,
}


def _prediction_component(conn: sqlite3.Connection, ticker: str, horizon_days: int) -> tuple[float, str] | None:
    """Return (score in [-1,1], description) from the latest stored prediction, or None if unavailable."""
    row = database.get_latest_prediction(conn, ticker, predictor.model_name_for(ticker, horizon_days))
    if row is None:
        return None
    prob_up = row["probability_up"]
    score = (prob_up - 0.5) * 2  # rescale 0..1 -> -1..1
    direction = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
    desc = f"ML model: {prob_up:.0%} probability of price rising over next {horizon_days}d ({direction})"
    return score, desc


def _sentiment_component(conn: sqlite3.Connection, ticker: str) -> tuple[float, str] | None:
    """Return (score in [-1,1], description) averaged over whichever of
    news/reddit/llm sentiment are available for the most recent date, or
    None if there's no sentiment data at all."""
    rows = database.get_latest_sentiment(conn, ticker)
    if not rows:
        return None
    scores = {row["source_type"]: row["score"] for row in rows}
    avg_score = sum(scores.values()) / len(scores)
    parts = ", ".join(f"{source}={score:+.2f}" for source, score in scores.items())
    direction = "positive" if avg_score > 0.1 else "negative" if avg_score < -0.1 else "mixed/neutral"
    desc = f"Sentiment ({parts}) — overall {direction}"
    return avg_score, desc


def _technical_component(conn: sqlite3.Connection, ticker: str, date: str) -> tuple[float, str] | None:
    """Return (score in [-1,1], description) from RSI (mean-reversion) and
    MACD (momentum) state on `date`, averaged over whichever are available.
    None if neither RSI nor MACD data exists for that date."""
    row = database.get_technical_indicators(conn, ticker, date)
    if row is None:
        return None

    sub_scores = []
    descs = []

    if row["rsi_14"] is not None:
        rsi = row["rsi_14"]
        # Mean-reversion read: RSI 50 is neutral; below 30 = oversold (bullish),
        # above 70 = overbought (bearish). Linear in between, clipped to [-1, 1].
        rsi_score = max(-1.0, min(1.0, (50 - rsi) / 30))
        sub_scores.append(rsi_score)
        state = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
        descs.append(f"RSI {rsi:.1f} ({state})")

    if row["macd"] is not None and row["macd_signal"] is not None:
        macd_score = 1.0 if row["macd"] > row["macd_signal"] else -1.0 if row["macd"] < row["macd_signal"] else 0.0
        sub_scores.append(macd_score)
        state = "bullish crossover" if macd_score > 0 else "bearish crossover" if macd_score < 0 else "flat"
        descs.append(f"MACD {state}")

    if not sub_scores:
        return None

    avg_score = sum(sub_scores) / len(sub_scores)
    return avg_score, "; ".join(descs)


def combine_components(components: dict[str, tuple[float, str]]) -> dict:
    """
    Shared decision logic: given whichever of {prediction, sentiment,
    technical} components are available (each a (score in [-1,1], description)
    tuple), compute the weighted composite, confidence, and BUY/SELL/HOLD
    call. Used by generate_signal() for live decisions AND by
    backtester.py for historical replay, so both paths make decisions the
    same way — the backtest is only meaningful if it's testing the same
    logic that runs live.

    Returns {signal, confidence, composite_score, reasoning}.
    Returns HOLD/confidence 0.0 if `components` is empty.
    """
    if not components:
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "composite_score": 0.0,
            "reasoning": "No prediction, sentiment, or technical data available yet — nothing to decide from.",
        }

    total_weight = sum(WEIGHTS[name] for name in components)
    composite = sum(WEIGHTS[name] * score for name, (score, _) in components.items()) / total_weight
    confidence = min(1.0, abs(composite))

    if confidence < config.MIN_CONFIDENCE_TO_ACT:
        signal = "HOLD"
    elif composite > 0:
        signal = "BUY"
    else:
        signal = "SELL"

    reasoning_parts = [desc for _, desc in components.values()]
    missing = [name for name in WEIGHTS if name not in components]
    if missing:
        reasoning_parts.append(f"(no {', '.join(missing)} data available — decision based on the rest)")
    if signal == "HOLD" and confidence < config.MIN_CONFIDENCE_TO_ACT and composite != 0:
        reasoning_parts.append(
            f"Composite score {composite:+.2f} did not clear the {config.MIN_CONFIDENCE_TO_ACT:.0%} confidence floor — defaulting to HOLD."
        )

    return {
        "signal": signal,
        "confidence": confidence,
        "composite_score": composite,
        "reasoning": " | ".join(reasoning_parts),
    }


def generate_signal(
    conn: sqlite3.Connection,
    ticker: str,
    horizon_days: int = predictor.DEFAULT_HORIZON_DAYS,
    persist: bool = True,
) -> dict:
    """
    Compute today's BUY/SELL/HOLD signal for `ticker` from whatever
    prediction/sentiment/technical data is currently available. Writes a row
    to the `signals` table (unless persist=False, used by backtester.py to
    replay historical decisions without polluting the live signal log).

    Returns a dict: {ticker, date, signal, confidence, reasoning, components}.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    latest_price_date = database.get_latest_price_date(conn, ticker)
    technical_date = latest_price_date or today

    components: dict[str, tuple[float, str]] = {}

    pred = _prediction_component(conn, ticker, horizon_days)
    if pred is not None:
        components["prediction"] = pred

    sent = _sentiment_component(conn, ticker)
    if sent is not None:
        components["sentiment"] = sent

    tech = _technical_component(conn, ticker, technical_date)
    if tech is not None:
        components["technical"] = tech

    decision = combine_components(components)

    result = {
        "ticker": ticker,
        "date": today,
        "signal": decision["signal"],
        "confidence": decision["confidence"],
        "reasoning": decision["reasoning"],
        "composite_score": decision["composite_score"],
        "components": {name: score for name, (score, _) in components.items()},
        "signal_id": None,
    }

    if persist:
        result["signal_id"] = database.insert_signal(
            conn, ticker, today, result["signal"], result["confidence"], result["reasoning"], dry_run=config.DRY_RUN
        )

    log.info("%s signal: %s (confidence=%.2f) — %s", ticker, result["signal"], result["confidence"], result["reasoning"])
    return result


def generate_watchlist_signals(
    conn: sqlite3.Connection, tickers: list[str] | None = None, horizon_days: int = predictor.DEFAULT_HORIZON_DAYS
) -> dict[str, dict]:
    tickers = tickers or config.WATCHLIST
    results: dict[str, dict] = {}
    for ticker in tickers:
        try:
            results[ticker] = generate_signal(conn, ticker, horizon_days)
        except Exception as exc:
            log.error("Unexpected error generating signal for %s: %s", ticker, exc)
            results[ticker] = {"ticker": ticker, "signal": "HOLD", "confidence": 0.0, "reasoning": f"error: {exc}"}
    return results


def generate_sma_vol_signal(conn: sqlite3.Connection, ticker: str) -> dict:
    """Generate an SMA(50/200) plus volatility-targeted position signal."""
    import pandas as pd
    import vol_target

    rows = database.get_price_history(conn, ticker)
    if not rows or len(rows) < 200:
        log.warning(
            "Not enough price history for SMA+vol signal on %s (%d rows).",
            ticker,
            len(rows) if rows else 0,
        )
        return {
            "ticker": ticker,
            "date": None,
            "regime": "unknown",
            "sma50": None,
            "sma200": None,
            "realized_vol": None,
            "target_vol": config.TARGET_VOL,
            "position_size": 0.0,
            "reasoning": "Insufficient price history for SMA(200).",
        }

    df = pd.DataFrame([dict(row) for row in rows])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["sma50"] = df["close"].rolling(window=50, min_periods=50).mean()
    df["sma200"] = df["close"].rolling(window=200, min_periods=200).mean()

    latest = df.iloc[-1]
    sma50 = float(latest["sma50"]) if pd.notna(latest["sma50"]) else None
    sma200 = float(latest["sma200"]) if pd.notna(latest["sma200"]) else None
    returns = df["close"].pct_change().dropna().iloc[-config.VOL_LOOKBACK_DAYS:]
    realized_vol = float(returns.std() * (252 ** 0.5)) if len(returns) > 0 else None

    if sma50 is None or sma200 is None:
        regime = "unknown"
        position_size = 0.0
    elif sma50 > sma200:
        regime = "bullish"
        position_size = vol_target.compute_position_size(
            df["close"], target_vol=config.TARGET_VOL, lookback=config.VOL_LOOKBACK_DAYS
        )
    else:
        regime = "bearish"
        position_size = 0.0

    latest_date = latest["date"].strftime("%Y-%m-%d")
    if regime == "bullish":
        reasoning = (
            f"SMA bullish (SMA50={sma50:.2f} > SMA200={sma200:.2f}). "
            f"Realized vol={realized_vol:.1%}, target vol={config.TARGET_VOL:.1%}. "
            f"Position size={position_size:.0%}."
        )
    elif regime == "bearish":
        reasoning = (
            f"SMA bearish (SMA50={sma50:.2f} < SMA200={sma200:.2f}). "
            "Position size=0% (cash)."
        )
    else:
        reasoning = "SMA values not yet available."

    log.info(
        "%s SMA+Vol: regime=%s position=%.0f%% — %s",
        ticker, regime, position_size * 100, reasoning,
    )
    return {
        "ticker": ticker,
        "date": latest_date,
        "regime": regime,
        "sma50": sma50,
        "sma200": sma200,
        "realized_vol": realized_vol,
        "target_vol": config.TARGET_VOL,
        "position_size": position_size,
        "reasoning": reasoning,
    }


def generate_sma_vol_watchlist(
    conn: sqlite3.Connection, tickers: list[str] | None = None
) -> dict[str, dict]:
    """Run the SMA plus volatility-targeted signal for every watchlist ticker."""
    tickers = tickers or config.WATCHLIST
    results: dict[str, dict] = {}
    for ticker in tickers:
        try:
            results[ticker] = generate_sma_vol_signal(conn, ticker)
        except Exception as exc:
            log.error("Unexpected error generating SMA+vol signal for %s: %s", ticker, exc)
            results[ticker] = {
                "ticker": ticker,
                "regime": "error",
                "position_size": 0.0,
                "reasoning": str(exc),
            }
    return results


if __name__ == "__main__":
    database.init_db()
    with database.get_connection() as conn:
        summary = generate_watchlist_signals(conn)
        log.info("=== Decision engine summary ===")
        for ticker, result in summary.items():
            log.info("%s: %s (confidence=%.2f)", ticker, result["signal"], result["confidence"])
