"""
llm_analyst.py — turns a decision_engine.py signal into a short, readable,
written thesis via an LLM (OpenAI, config.LLM_MODEL_DEEP — "gpt-4o" for
deep reasoning, per the tech stack).

This module does NOT make decisions. decision_engine.py has already decided
BUY/SELL/HOLD and computed confidence before this module ever runs; all
llm_analyst.py does is explain that decision in prose a human can read
quickly, grounded in the same numbers decision_engine used. If the LLM's
prose ever seems to imply a different call than the signal it was given,
that's a prompting bug to fix, not a second opinion to trust.

Hard rule enforced here (from the project brief): "Never claim guaranteed
profit — output is analysis, not financial advice." This is stated
explicitly in the system prompt below, and every thesis is checked for
some obviously disqualifying language patterns before being stored —
belt-and-suspenders, not a substitute for the prompt doing its job.

Best-effort like every other module: no config.OPENAI_API_KEY or no
`openai` package installed means this is skipped with a warning, and the
signal that decision_engine.py already wrote and persisted is left exactly
as it is — a missing thesis is a missing convenience, not a missing signal.

By default, a thesis is only generated for signals in config.THESIS_SIGNALS
(BUY/SELL) to avoid spending an LLM call explaining a no-conviction HOLD.
"""

from __future__ import annotations

import sqlite3

import config
import database

log = config.get_logger(__name__)

SYSTEM_PROMPT = """You are a financial research assistant writing a short
analyst note. You are explaining a trading signal that has ALREADY been
computed by a separate quantitative system — you are not deciding it, you
are narrating the reasoning behind a decision that is given to you.

Hard rules, no exceptions:
- Never claim or imply a guaranteed, certain, or "sure thing" outcome.
- This is analysis, not financial advice. Do not tell the reader what they
  personally should do with their money.
- Do not invent facts, numbers, or events not present in the data you were given.
- Stay grounded in the specific numbers provided — do not use generic
  boilerplate that could apply to any stock.
- Keep it to 2-4 short paragraphs.
"""

# Extremely crude, deliberately conservative post-hoc check. This is a
# backstop, not the primary safeguard (the system prompt above is) — if this
# ever fires, treat it as a signal the prompt needs tightening, not just a
# string to filter out.
_DISQUALIFYING_PHRASES = [
    "guaranteed", "guarantee", "sure thing", "cannot lose", "can't lose",
    "risk-free", "will definitely", "certain to",
]


def _build_user_prompt(ticker: str, signal: dict, context: dict) -> str:
    return f"""Ticker: {ticker}
Signal: {signal["signal"]} (confidence: {signal["confidence"]:.0%})
Structured reasoning from the decision engine: {signal["reasoning"]}

Supporting data:
{context["formatted"]}

Write a short analyst note explaining why the decision engine likely
reached this {signal["signal"]} call, referencing the specific numbers
above. End with a one-sentence reminder that this is automated analysis,
not financial advice, and that DRY_RUN mode means no trade is being placed.
"""


def _gather_context(conn: sqlite3.Connection, ticker: str) -> dict:
    """Pull the same kind of numbers decision_engine.py used, formatted as
    plain text for the prompt. Doesn't need to be exhaustive — just grounded."""
    lines = []

    latest_date = database.get_latest_price_date(conn, ticker)
    if latest_date:
        price_row = conn.execute(
            "SELECT close FROM prices WHERE ticker = ? AND date = ?", (ticker, latest_date)
        ).fetchone()
        if price_row:
            lines.append(f"- Latest close ({latest_date}): ${price_row['close']:.2f}")

        tech_row = database.get_technical_indicators(conn, ticker, latest_date)
        if tech_row:
            if tech_row["rsi_14"] is not None:
                lines.append(f"- RSI(14): {tech_row['rsi_14']:.1f}")
            if tech_row["macd"] is not None and tech_row["macd_signal"] is not None:
                lines.append(f"- MACD: {tech_row['macd']:.3f} vs signal {tech_row['macd_signal']:.3f}")
            if tech_row["sma_50"] is not None:
                lines.append(f"- SMA(50): ${tech_row['sma_50']:.2f}")
            if tech_row["sma_200"] is not None:
                lines.append(f"- SMA(200): ${tech_row['sma_200']:.2f}")

    sentiment_rows = database.get_latest_sentiment(conn, ticker)
    for row in sentiment_rows:
        lines.append(f"- {row['source_type'].capitalize()} sentiment: {row['score']:+.2f} ({row['label']})")

    if not lines:
        lines.append("(no supporting technical/sentiment data was available)")

    return {"formatted": "\n".join(lines)}


def _passes_safety_check(text: str) -> bool:
    lowered = text.lower()
    return not any(phrase in lowered for phrase in _DISQUALIFYING_PHRASES)


def generate_thesis(conn: sqlite3.Connection, ticker: str, signal: dict) -> str | None:
    """
    Generate a written thesis for an already-computed `signal` dict (as
    returned by decision_engine.generate_signal). Returns the thesis text,
    or None if OPENAI_API_KEY/openai aren't available, the call failed, or
    the generated text failed the safety check.
    """
    if not config.OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY not set — skipping thesis generation for %s.", ticker)
        return None

    try:
        from openai import OpenAI
    except ImportError:
        log.warning("openai package not installed — skipping thesis generation for %s. `pip install openai`.", ticker)
        return None

    context = _gather_context(conn, ticker)
    user_prompt = _build_user_prompt(ticker, signal, context)

    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.LLM_MODEL_DEEP,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        thesis = response.choices[0].message.content.strip()
    except Exception as exc:
        log.error("LLM thesis generation failed for %s: %s", ticker, exc)
        return None

    if not _passes_safety_check(thesis):
        log.error(
            "Generated thesis for %s failed the safety check (disqualifying language) — discarding it. "
            "This should not happen given the system prompt; treat it as a prompt bug.",
            ticker,
        )
        return None

    return thesis


def generate_and_store_thesis(conn: sqlite3.Connection, ticker: str, signal: dict) -> str | None:
    """
    Generate a thesis for `signal` (as returned by decision_engine.generate_signal,
    with persist=True so it has a signal_id) and attach it to that exact
    signal row. Only bothers for signals in config.THESIS_SIGNALS (BUY/SELL
    by default) — skips HOLD to save the cost of an LLM call explaining
    "nothing to do right now."

    Returns the thesis text, or None if generation was skipped/failed.
    """
    if signal["signal"] not in config.THESIS_SIGNALS:
        log.info("%s: signal is %s, not in THESIS_SIGNALS %s — skipping thesis.", ticker, signal["signal"], config.THESIS_SIGNALS)
        return None

    if signal.get("signal_id") is None:
        log.warning("%s: signal has no signal_id (was it generated with persist=False?) — cannot attach a thesis.", ticker)
        return None

    thesis = generate_thesis(conn, ticker, signal)
    if thesis is None:
        return None

    database.update_signal_thesis(conn, signal["signal_id"], thesis)
    log.info("Stored thesis for %s (signal_id=%d, %d chars).", ticker, signal["signal_id"], len(thesis))
    return thesis


if __name__ == "__main__":
    import decision_engine

    database.init_db()
    with database.get_connection() as conn:
        for ticker in config.WATCHLIST:
            sig = decision_engine.generate_signal(conn, ticker)
            thesis = generate_and_store_thesis(conn, ticker, sig)
            log.info("%s: %s — thesis %s", ticker, sig["signal"], "generated" if thesis else "skipped")
