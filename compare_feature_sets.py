"""Compare predictive/backtest performance across feature groups."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import backtester
import config
import database
import macro_fetcher
import predictor

RESULTS_PATH = Path(__file__).resolve().parent / "backtests" / "feature_comparison.json"


def feature_sets() -> dict[str, list[str]]:
    original = predictor.ORIGINAL_FEATURE_COLUMNS
    return {
        "original": original,
        "original_plus_fundamentals": original + predictor.FUNDAMENTAL_FEATURE_COLUMNS,
        "original_plus_sentiment": original + predictor.SENTIMENT_AGGREGATE_FEATURE_COLUMNS,
        "original_plus_macro": original + predictor.MACRO_FEATURE_COLUMNS,
        "all_combined": predictor.FEATURE_COLUMNS,
    }


def main() -> None:
    database.init_db()
    rows: list[dict] = []
    with database.get_connection() as conn:
        macro_fetcher.fetch_all_macro(conn, date.today() - timedelta(days=365 * 5))
        for ticker in config.WATCHLIST:
            for name, columns in feature_sets().items():
                metrics = backtester.run_walkforward_backtest(
                    conn, ticker, feature_columns=columns
                )
                if metrics is None:
                    continue
                rows.append({
                    "ticker": ticker,
                    "feature_set": name,
                    "holdout_accuracy": metrics.get("holdout_accuracy"),
                    "backtest_sharpe": metrics.get("sharpe_ratio"),
                    "trades": metrics.get("n_trades", 0),
                    "feature_importances": metrics.get("feature_importances", {}),
                })

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as file:
        json.dump(rows, file, indent=2, default=str)

    header = f"{'ticker':<8}{'feature_set':<30}{'holdout_accuracy':>18}{'backtest_sharpe':>18}{'trades':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        accuracy = "n/a" if row["holdout_accuracy"] is None else f"{row['holdout_accuracy']:.3f}"
        sharpe = "n/a" if row["backtest_sharpe"] is None else f"{row['backtest_sharpe']:.3f}"
        print(f"{row['ticker']:<8}{row['feature_set']:<30}{accuracy:>18}{sharpe:>18}{row['trades']:>10}")
    print(f"\nSaved {len(rows)} comparison rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
