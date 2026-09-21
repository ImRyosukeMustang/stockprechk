"""
predictor.py — machine-learning model predicting P(price up) over a forward
horizon, from technical + sentiment features.
Model: XGBoost (per the tech stack) if installed, otherwise a fallback to
scikit-learn's HistGradientBoostingClassifier — chosen deliberately because,
like XGBoost, it handles NaN feature values natively (important here: early
dates lack sma_200, and sentiment coverage is sparse — see module docstring
in sentiment.py). Whichever backend is used, this module exposes the same
train/predict interface, and every saved model records which backend
produced it so predictions are never silently mixed across incompatible
model files.
No look-ahead bias:
  - Every feature at row (ticker, date) is built ONLY from that date's price
    and technical-indicator values (both already trailing-only by
    construction — see technical.py) and same-day sentiment.
  - The label at row (ticker, date) is whether close price horizon_days
    trading days LATER is higher than close price at date. Using future
    data to construct a *label* for supervised learning is standard and
    correct; the discipline that matters is that the *features* at that row
    never include anything from after date. This module enforces that by
    construction — it never joins forward-looking columns into the feature
    set.
  - The last horizon_days rows of any ticker's history have no valid label
    (we don't yet know the future close) and are dropped from training.
Known limitation carried over from sentiment.py: because sentiment_scores
only has rows for recent dates, sentiment features will be NaN/missing for
most of history. They're included in the feature set so the model can use
them when present, but the model should not be expected to lean on them
until historical sentiment coverage improves.
This module does NOT decide trading logic — it only produces a probability.
decision_engine.py (next file) combines this probability with sentiment and
technical readings into an actual BUY/SELL/HOLD call.
Dtype safety:
  XGBoost (and sklearn's HistGradientBoostingClassifier) reject pandas
  DataFrames containing object / nullable-extension dtypes. Two defensive
  layers keep the feature matrix clean:
    1. build_feature_frame() forces the sentiment columns to float64
       immediately after the pivot/merge, so they can never become
       pure-object and thus never get silently dropped.
    2. _sanitize_features() reindexes the matrix to exactly the trained
       feature list (creating NaNs for any missing column) and coerces
       everything to float64, replacing ±inf with NaN along the way.
"""
from **future** import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import config
import database
log = config.get_logger(**name**)
MODELS_DIR = Path(**file**).resolve().parent / "models"
DEFAULT_HORIZON_DAYS = 5
MIN_TRAINING_ROWS = 60  # below this, a walk-forward split is meaningless
FEATURE_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_percent",
    "close_to_sma50",
    "close_to_sma200",
    "daily_return",
    "volume_change",
    "sentiment_news",
    "sentiment_reddit",
    "sentiment_llm",
]
# ---------------------------------------------------------------------------
# Dtype hygiene
# ---------------------------------------------------------------------------
def _sanitize_features(X, expected_columns: list[str] | None = None):
    """
    Force a clean numeric matrix for XGBoost / sklearn.
    - If expected_columns is given, reindex X to exactly those columns (in
      that order). Any missing column is created as NaN and a warning is
      logged, so train/predict always see the same feature shape.
    - Coerce every column to float64, killing object / nullable-extension
      dtypes that XGBoost's pandas mapper can't handle.
    - Replace ±inf with NaN (XGBoost handles NaN natively; inf crashes it).
    """
    import numpy as np
    import pandas as pd
    if not isinstance(X, pd.DataFrame):
        return X
    if expected_columns is not None:
        missing = [c for c in expected_columns if c not in X.columns]
        if missing:
            log.warning("Feature matrix missing expected columns: %s", missing)
            for c in missing:
                X[c] = np.nan
        # Keep only the expected columns, in the exact order the model was trained with
        X = X.reindex(columns=expected_columns)
    # Force pure float64 — this is the key step that kills object /
    # extension dtypes.
    X = X.apply(pd.to_numeric, errors="coerce").astype("float64")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X
# ---------------------------------------------------------------------------
# Model naming / paths
# ---------------------------------------------------------------------------
def _model_name(ticker: str, horizon_days: int) -> str:
    return f"{ticker.upper()}_h{horizon_days}"
def model_name_for(ticker: str, horizon_days: int) -> str:
    """Public helper so other modules (decision_engine.py) can look up a
    ticker's prediction rows via database.get_latest_prediction without
    duplicating the naming convention."""
    return _model_name(ticker, horizon_days)
def _model_path(ticker: str, horizon_days: int) -> Path:
    return MODELS_DIR / f"{_model_name(ticker, horizon_days)}.joblib"
# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def build_feature_frame(conn: sqlite3.Connection, ticker: str):
    """
    Build a feature DataFrame (one row per date) for ticker from stored
    price/indicator/sentiment data. Returns None if pandas isn't installed
    or there isn't any price history yet.
    """
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not installed — predictor cannot build features for %s.", ticker)
        return None
    rows = database.get_price_and_indicators(conn, ticker)
    if not rows:
        log.warning("No price history stored for %s — cannot build features.", ticker)
        return None
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # Derived features from stored raw indicators.
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    bb_width = df["bb_upper"] - df["bb_lower"]
    df["bb_percent"] = (df["close"] - df["bb_lower"]) / bb_width  # 0 = at lower band, 1 = at upper band
    df.loc[bb_width == 0, "bb_percent"] = None
    df["close_to_sma50"] = (df["close"] - df["sma_50"]) / df["sma_50"]
    df["close_to_sma200"] = (df["close"] - df["sma_200"]) / df["sma_200"]
    df["daily_return"] = df["close"].pct_change()
    df["volume_change"] = df["volume"].pct_change()
    # Sentiment is a separate, sparse table — left-join it in per date.
    sentiment_rows = conn.execute(
        "SELECT date, source_type, score FROM sentiment_scores WHERE ticker = ?", (ticker,)
    ).fetchall()
    if sentiment_rows:
        sent_df = pd.DataFrame([dict(r) for r in sentiment_rows])
        sent_df["date"] = pd.to_datetime(sent_df["date"])
        pivot = sent_df.pivot_table(index="date", columns="source_type", values="score", aggfunc="mean")
        pivot = pivot.rename(columns={"news": "sentiment_news", "reddit": "sentiment_reddit", "llm": "sentiment_llm"})
        df = df.merge(pivot, on="date", how="left")
    # Force sentiment columns to exist and be float64 — never let them become
    # pure-object, which would cause _sanitize_features to drop them.
    for col in ("sentiment_news", "sentiment_reddit", "sentiment_llm"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    # Same hardening for the numeric price-derived features, in case any
    # pct_change/NaN arithmetic left them as object.
    for col in (
        "rsi_14", "macd", "macd_signal", "macd_hist", "bb_percent",
        "close_to_sma50", "close_to_sma200", "daily_return", "volume_change",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df
# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def _get_model_backend():
    """Return (backend_name, model_factory) — tries XGBoost first, falls
    back to sklearn's HistGradientBoostingClassifier (also handles NaN
    natively, matching XGBoost's behavior on missing indicator/sentiment
    values)."""
    try:
        import xgboost as xgb
        def factory():
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
            )
        return "xgboost", factory
    except ImportError:
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        def factory():
            return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200)
        return "sklearn_hgb", factory
    except ImportError:
        return None, None
# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(conn: sqlite3.Connection, ticker: str, horizon_days: int = DEFAULT_HORIZON_DAYS) -> dict | None:
    """
    Train (or retrain) a probability-up model for ticker. Reports a quick
    time-ordered holdout accuracy for diagnostics, then fits a deployment
    model on the full available history and saves it to disk.
    Returns a summary dict, or None if training couldn't proceed (missing
    dependency, insufficient data).
    """
    try:
        import joblib
    except ImportError:
        log.warning("joblib not installed — cannot save trained models. pip install joblib.")
        return None
    backend_name, factory = _get_model_backend()
    if factory is None:
        log.warning(
            "Neither xgboost nor scikit-learn is installed — cannot train a model for %s.", ticker
        )
        return None
    df = build_feature_frame(conn, ticker)
    if df is None:
        return None
    # Label: did close horizon_days trading days later end up higher?
    df["future_close"] = df["close"].shift(-horizon_days)
    df["label"] = (df["future_close"] > df["close"]).astype(int)
    # Rows with no known future close (the tail of history) can't be trained on.
    df = df.iloc[:-horizon_days] if horizon_days > 0 else df
    if len(df) < MIN_TRAINING_ROWS:
        log.warning(
            "Only %d usable rows for %s (need >= %d) — skipping training. "
            "Fetch more price history first (config.PRICE_HISTORY_PERIOD).",
            len(df), ticker, MIN_TRAINING_ROWS,
        )
        return None
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    # Sanitize once — the same cleaned matrix is reused for holdout + final fit.
    X = _sanitize_features(X, expected_columns=FEATURE_COLUMNS)
    # Quick time-ordered holdout (last 20%) purely for a diagnostic accuracy
    # number — this is NOT the walk-forward backtest (that's backtester.py,
    # which retrains across multiple folds). This is a cheap sanity check.
    split_idx = int(len(df) * 0.8)
    X_train, X_holdout = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_holdout = y.iloc[:split_idx], y.iloc[split_idx:]
    holdout_accuracy = None
    if len(X_holdout) > 0 and y_holdout.nunique() > 1:
        diag_model = factory()
        diag_model.fit(X_train, y_train)
        holdout_accuracy = float(diag_model.score(X_holdout, y_holdout))
        log.info(
            "%s: holdout accuracy over last %d rows = %.3f (backend=%s)",
            ticker, len(X_holdout), holdout_accuracy, backend_name,
        )
    else:
        log.warning(
            "%s: holdout set too small or single-class — skipping diagnostic accuracy.", ticker
        )
    # Deployment model: trained on the full dataset, saved to disk.
    final_model = factory()
    final_model.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = _model_path(ticker, horizon_days)
    joblib.dump(
        {
            "backend": backend_name,
            "ticker": ticker,
            "horizon_days": horizon_days,
            "feature_columns": FEATURE_COLUMNS,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_rows_trained": len(df),
            "holdout_accuracy": holdout_accuracy,
            "model": final_model,
        },
        path,
    )
    log.info("Saved trained model for %s to %s (backend=%s, rows=%d)", ticker, path, backend_name, len(df))
    return {
        "ticker": ticker,
        "horizon_days": horizon_days,
        "backend": backend_name,
        "n_rows_trained": len(df),
        "holdout_accuracy": holdout_accuracy,
        "model_path": str(path),
    }
# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_latest(conn: sqlite3.Connection, ticker: str, horizon_days: int = DEFAULT_HORIZON_DAYS) -> float | None:
    """
    Load the saved model for ticker/horizon_days and predict P(up) for
    the most recent available date. Stores the prediction in the
    predictions table. Returns the probability, or None if no model has
    been trained yet or a dependency is missing.
    """
    try:
        import joblib
    except ImportError:
        log.warning("joblib not installed — cannot load trained models.")
        return None
    path = _model_path(ticker, horizon_days)
    if not path.exists():
        log.warning(
            "No trained model found for %s at horizon=%d (%s). Call train_model() first.",
            ticker, horizon_days, path,
        )
        return None
    bundle = joblib.load(path)
    df = build_feature_frame(conn, ticker)
    if df is None or df.empty:
        log.warning("No feature data available for %s — cannot predict.", ticker)
        return None
    latest_row = df.iloc[[-1]]
    X_latest = latest_row[bundle["feature_columns"]]
    X_latest = _sanitize_features(X_latest, expected_columns=bundle["feature_columns"])
    try:
        probability_up = float(bundle["model"].predict_proba(X_latest)[0][1])
    except Exception as exc:
        log.error("Prediction failed for %s: %s", ticker, exc)
        return None
    latest_date = latest_row["date"].iloc[0].strftime("%Y-%m-%d")
    model_name = _model_name(ticker, horizon_days)
    database.insert_prediction(conn, ticker, latest_date, model_name, probability_up)
    log.info("Predicted P(up over %dd) for %s on %s: %.3f", horizon_days, ticker, latest_date, probability_up)
    return probability_up
# ---------------------------------------------------------------------------
# Watchlist driver
# ---------------------------------------------------------------------------
def train_and_predict_watchlist(
    conn: sqlite3.Connection, tickers: list[str] | None = None, horizon_days: int = DEFAULT_HORIZON_DAYS
) -> dict[str, float | None]:
    """Train (or retrain) and predict for every ticker in the watchlist. One
    ticker's failure doesn't stop the others."""
    tickers = tickers or config.WATCHLIST
    results: dict[str, float | None] = {}
    for ticker in tickers:
        try:
            train_summary = train_model(conn, ticker, horizon_days)
            if train_summary is None:
                results[ticker] = None
                continue
            results[ticker] = predict_latest(conn, ticker, horizon_days)
        except Exception as exc:
            log.error("Unexpected error training/predicting for %s: %s", ticker, exc)
            results[ticker] = None
    return results
if **name** == "**main**":
    database.init_db()
    with database.get_connection() as conn:
        summary = train_and_predict_watchlist(conn)
        log.info("=== Predictor summary ===")
        for ticker, prob in summary.items():
            log.info("%s: P(up) = %s", ticker, prob)
