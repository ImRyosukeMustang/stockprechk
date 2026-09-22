"""
config.py — StockOracle central configuration.

All API keys, tunables, and safety switches live here. Every other module
imports from this file rather than reading environment variables directly,
so there is exactly one place to look when something needs to change.

Nothing in this file talks to the network or the database. It only defines
values and does light validation/logging on import.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# .env loading (optional — falls back to real environment variables if
# python-dotenv isn't installed or there is no .env file yet)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    # python-dotenv not installed yet — that's fine, os.environ still works.
    pass

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# SAFETY SWITCH — the whole project must respect this. No module should ever
# place a real order, submit a real trade, or otherwise touch real money
# while DRY_RUN is True. This starts True and should only be flipped by a
# human, after backtesting, never automatically by code.
# ---------------------------------------------------------------------------
DRY_RUN: bool = True

# ---------------------------------------------------------------------------
# API keys — read from Streamlit secrets (cloud) OR environment (local).
# All are optional at Day 0; individual modules are responsible for checking
# whether a key they need is present and failing gracefully (skip that data
# source, log a warning) rather than crashing the whole pipeline.
# ---------------------------------------------------------------------------
def _get_secret(key: str, default: str | None = None) -> str | None:
    """Read a secret from Streamlit's secrets system if running under
    Streamlit Cloud, otherwise from os.environ (local development)."""
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        # Not running under Streamlit, or secrets not configured
        pass
    return os.environ.get(key, default)


OPENAI_API_KEY: str | None = _get_secret("OPENAI_API_KEY")
NEWSAPI_KEY: str | None = _get_secret("NEWSAPI_KEY")
REDDIT_CLIENT_ID: str | None = _get_secret("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET: str | None = _get_secret("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT: str = _get_secret("REDDIT_USER_AGENT", "StockOracle/0.1")

# ---------------------------------------------------------------------------
# LLM model selection
# ---------------------------------------------------------------------------
LLM_MODEL_CHEAP: str = "gpt-4o-mini"   # sentiment tagging, quick classification
LLM_MODEL_DEEP: str = "gpt-4o"         # full thesis writing, deep reasoning

# ---------------------------------------------------------------------------
# Watchlist — the universe of tickers the pipeline operates on. Day 0: a
# small, deliberately boring starter list. Edit freely.
# ---------------------------------------------------------------------------
WATCHLIST: list[str] = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
]

# ---------------------------------------------------------------------------
# Risk / decision-engine settings (Phase 3 will consume these; defined now
# so config.py is the single source of truth from day one)
# ---------------------------------------------------------------------------
MAX_POSITION_PCT: float = 0.05      # max % of a hypothetical portfolio per position
MIN_CONFIDENCE_TO_ACT: float = 0.60  # signals below this confidence become HOLD
STOP_LOSS_PCT: float = 0.08
TAKE_PROFIT_PCT: float = 0.15

# ---------------------------------------------------------------------------
# Predictor (Phase 3 — the ML brain)
# ---------------------------------------------------------------------------
MODELS_DIR: Path = BASE_DIR / "models"
MODEL_NAME: str = "xgboost_v1"          # logical name stored alongside predictions
MIN_TRAINING_SAMPLES: int = 60          # fewer rows than this and we refuse to train
PREDICTION_HORIZON_DAYS: int = 1        # predict direction N trading days ahead

# ---------------------------------------------------------------------------
# Backtester (Phase 3 — walk-forward validation, never a random split)
# ---------------------------------------------------------------------------
TRANSACTION_COST_PCT: float = 0.001     # 0.1% per trade (both entry and exit), a conservative
                                         # round-trip-friction estimate for a liquid large-cap stock
WALK_FORWARD_TRAIN_WINDOW: int = 120    # trading days used to train before each test block
WALK_FORWARD_TEST_WINDOW: int = 20      # trading days evaluated before retraining and rolling forward
WALK_FORWARD_MIN_TOTAL_BARS: int = WALK_FORWARD_TRAIN_WINDOW + WALK_FORWARD_TEST_WINDOW
TRANSACTION_COST_PCT: float = 0.001  # 10 bps per trade side (commission + slippage estimate)

# ---------------------------------------------------------------------------
# Data-fetch settings
# ---------------------------------------------------------------------------
PRICE_HISTORY_PERIOD: str = "5y"     # yfinance period string
PRICE_HISTORY_INTERVAL: str = "1d"   # yfinance interval string
NEWS_LOOKBACK_HOURS: int = 48
REDDIT_SUBREDDITS: list[str] = ["stocks", "wallstreetbets", "investing"]
REDDIT_POST_LIMIT: int = 25

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
DB_PATH: Path = BASE_DIR / "stockoracle.db"

# ---------------------------------------------------------------------------
# Scheduler (Phase 4)
# ---------------------------------------------------------------------------
PIPELINE_INTERVAL_MINUTES: int = 60

# ---------------------------------------------------------------------------
# LLM Analyst (Phase 4 — written thesis)
# ---------------------------------------------------------------------------
# Writing a full thesis costs a real gpt-4o call per ticker, so by default we
# only bother for signals worth explaining to a human — HOLD with no
# conviction doesn't need a paragraph justifying it. Override freely.
THESIS_SIGNALS: list[str] = ["BUY", "SELL"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("STOCKORACLE_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured consistently across the project."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(LOG_LEVEL)
    return logger


_log = get_logger(__name__)

if DRY_RUN:
    _log.info("StockOracle running in DRY_RUN mode — no real trades will be placed.")
else:
    _log.warning(
        "DRY_RUN is False. This should only happen after backtesting has "
        "validated the strategy. Proceeding with live-mode assumptions."
    )

# Warn (don't crash) about missing keys so Phase 1 data fetching can still
# run partially against whatever sources are configured.
for _key_name, _key_val in (
    ("OPENAI_API_KEY", OPENAI_API_KEY),
    ("NEWSAPI_KEY", NEWSAPI_KEY),
    ("REDDIT_CLIENT_ID", REDDIT_CLIENT_ID),
    ("REDDIT_CLIENT_SECRET", REDDIT_CLIENT_SECRET),
):
    if not _key_val:
        _log.warning("%s is not set — features depending on it will be skipped.", _key_name)
