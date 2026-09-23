from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import config
import database

logger = config.get_logger(__name__)

DB_PATH = "stockoracle.db"
OUTPUT_PATH = Path("backtests/conditional_edge_results.json")

FORWARD_DAYS = 5
SMA_SHORT = 50
SMA_LONG = 200

BUCKET_EDGES = [0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
BUCKET_LABELS = ["0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70+"]

MIN_ROWS_REQUIRED = SMA_LONG + FORWARD_DAYS + 1


def load_price_df(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame | None:
    """Load price history for a ticker and return a sorted DataFrame, or None if unusable."""
    try:
        rows = database.get_price_history(conn, ticker)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load price history for %s: %s", ticker, exc)
        return None

    if not rows:
        logger.warning("No price history rows found for %s", ticker)
        return None

    df = pd.DataFrame([dict(r) for r in rows])

    if "close" not in df.columns or "date" not in df.columns:
        logger.warning("Price history for %s missing required columns (date/close)", ticker)
        return None

    if len(df) < MIN_ROWS_REQUIRED:
        logger.warning(
            "Insufficient price history for %s: %d rows (need at least %d)",
            ticker,
            len(df),
            MIN_ROWS_REQUIRED,
        )
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ticker"] = ticker
    return df


def compute_smas(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA(50) and SMA(200) columns computed from close price."""
    df = df.copy()
    df["sma50"] = df["close"].rolling(window=SMA_SHORT, min_periods=SMA_SHORT).mean()
    df["sma200"] = df["close"].rolling(window=SMA_LONG, min_periods=SMA_LONG).mean()
    return df


def compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add forward N-day return column; rows without a future close get NaN."""
    df = df.copy()
    future_close = df["close"].shift(-FORWARD_DAYS)
    df["forward_return"] = (future_close - df["close"]) / df["close"] * 100.0
    return df


def load_predictions(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame | None:
    """Load predictions for a ticker, keeping only the most recent matching model."""
    model_pattern = f"{ticker}_h5"
    try:
        query = (
            "SELECT ticker, date, model_name, probability_up "
            "FROM predictions "
            "WHERE ticker = ? AND model_name LIKE ?"
        )
        pred_df = pd.read_sql_query(query, conn, params=(ticker, f"{model_pattern}%"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load predictions for %s: %s", ticker, exc)
        return None

    if pred_df.empty:
        logger.warning("No predictions found for %s matching pattern '%s%%'", ticker, model_pattern)
        return None

    pred_df["date"] = pd.to_datetime(pred_df["date"])

    # If multiple model versions exist under the pattern, keep only the most
    # recent model_name (determined by that model's latest prediction date).
    if pred_df["model_name"].nunique() > 1:
        latest_dates = pred_df.groupby("model_name")["date"].max()
        most_recent_model = latest_dates.idxmax()
        pred_df = pred_df[pred_df["model_name"] == most_recent_model]
        logger.info(
            "Multiple model versions found for %s; using most recent: %s",
            ticker,
            most_recent_model,
        )

    return pred_df[["ticker", "date", "probability_up"]]


def build_bullish_regime_df(
    price_df: pd.DataFrame, pred_df: pd.DataFrame
) -> pd.DataFrame | None:
    """Merge price + predictions, restrict to bullish SMA regime with valid forward returns."""
    merged = pd.merge(price_df, pred_df, on=["ticker", "date"], how="inner")

    # Require both SMAs defined
    merged = merged.dropna(subset=["sma50", "sma200"])

    # Bullish regime filter
    merged = merged[merged["sma50"] > merged["sma200"]]

    # Drop rows without a valid forward return (near the end of the series)
    merged = merged.dropna(subset=["forward_return"])

    # Only interested in the bullish-probability buckets (>= 0.50)
    merged = merged[merged["probability_up"] >= 0.50]

    if merged.empty:
        return None

    return merged


def compute_bucket_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket probability_up into 5 bins and compute summary stats per bucket."""
    df = df.copy()
    df["bucket"] = pd.cut(
        df["probability_up"],
        bins=BUCKET_EDGES,
        labels=BUCKET_LABELS,
        right=False,
        include_lowest=True,
    )

    stats_rows = []
    for label in BUCKET_LABELS:
        bucket_df = df[df["bucket"] == label]
        n = len(bucket_df)
        if n == 0:
            stats_rows.append(
                {
                    "bucket": label,
                    "avg_return": np.nan,
                    "median_return": np.nan,
                    "hit_rate": np.nan,
                    "n": 0,
                }
            )
            continue

        avg_return = bucket_df["forward_return"].mean()
        median_return = bucket_df["forward_return"].median()
        hit_rate = (bucket_df["forward_return"] > 0).mean() * 100.0

        stats_rows.append(
            {
                "bucket": label,
                "avg_return": avg_return,
                "median_return": median_return,
                "hit_rate": hit_rate,
                "n": n,
            }
        )

    return pd.DataFrame(stats_rows)


def format_bucket_table(stats_df: pd.DataFrame, title: str) -> str:
    """Render the bucket stats table as a printable string."""
    lines = [f"=== {title} ==="]
    lines.append(f"{'Bucket':<12}{'Avg 5d':>10}{'Median 5d':>12}{'Hit%':>8}{'N':>8}")

    for _, row in stats_df.iterrows():
        if row["n"] == 0:
            lines.append(f"{row['bucket']:<12}{'--':>10}{'--':>12}{'--':>8}{0:>8}")
            continue

        avg_str = f"{row['avg_return']:+.2f}%"
        median_str = f"{row['median_return']:+.2f}%"
        hit_str = f"{row['hit_rate']:.0f}%"
        lines.append(
            f"{row['bucket']:<12}{avg_str:>10}{median_str:>12}{hit_str:>8}{row['n']:>8}"
        )

    return "\n".join(lines)


def is_monotonic_increasing(stats_df: pd.DataFrame) -> bool:
    """Check whether avg_return increases monotonically across buckets with data."""
    valid = stats_df.dropna(subset=["avg_return"])
    if len(valid) < 2:
        return False

    avg_returns = valid["avg_return"].to_numpy()
    diffs = np.diff(avg_returns)
    return bool(np.all(diffs > 0))


def stats_df_to_records(stats_df: pd.DataFrame) -> list[dict]:
    """Convert a bucket stats DataFrame into JSON-serializable records."""
    records = []
    for _, row in stats_df.iterrows():
        records.append(
            {
                "bucket": row["bucket"],
                "avg_return": None if pd.isna(row["avg_return"]) else round(float(row["avg_return"]), 4),
                "median_return": None if pd.isna(row["median_return"]) else round(float(row["median_return"]), 4),
                "hit_rate": None if pd.isna(row["hit_rate"]) else round(float(row["hit_rate"]), 2),
                "n": int(row["n"]),
            }
        )
    return records


def main() -> None:
    results: dict[str, object] = {"tickers": {}, "aggregate": {}}
    all_regime_dfs: list[pd.DataFrame] = []

    with database.get_connection() as conn:
        for ticker in config.WATCHLIST:
            logger.info("Processing %s", ticker)

            price_df = load_price_df(conn, ticker)
            if price_df is None:
                continue

            price_df = compute_smas(price_df)
            price_df = compute_forward_returns(price_df)

            pred_df = load_predictions(conn, ticker)
            if pred_df is None:
                continue

            regime_df = build_bullish_regime_df(price_df, pred_df)
            if regime_df is None or regime_df.empty:
                logger.warning("No bullish-regime rows with predictions for %s; skipping", ticker)
                continue

            stats_df = compute_bucket_stats(regime_df)
            table = format_bucket_table(stats_df, f"{ticker} (bullish regime)")
            print(table)
            print()

            results["tickers"][ticker] = {
                "buckets": stats_df_to_records(stats_df),
                "total_samples": int(len(regime_df)),
            }

            all_regime_dfs.append(regime_df)

    if not all_regime_dfs:
        logger.error("No usable data for any ticker. Cannot compute aggregate or verdict.")
        print("No usable data across any ticker in the watchlist. Aborting.")
        return

    combined_df = pd.concat(all_regime_dfs, ignore_index=True)
    aggregate_stats_df = compute_bucket_stats(combined_df)
    aggregate_table = format_bucket_table(aggregate_stats_df, "AGGREGATE (all tickers, bullish regime)")
    print(aggregate_table)
    print()

    results["aggregate"] = {
        "buckets": stats_df_to_records(aggregate_stats_df),
        "total_samples": int(len(combined_df)),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", OUTPUT_PATH)

    if is_monotonic_increasing(aggregate_stats_df):
        print("✅ ML HAS CONDITIONAL EDGE. Build the hybrid.")
    else:
        print("❌ ML HAS NO CONDITIONAL EDGE. Skip the hybrid. Use SMA alone.")


if __name__ == "__main__":
    main()
