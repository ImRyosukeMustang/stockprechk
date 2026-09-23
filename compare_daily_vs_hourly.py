from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config

log = config.get_logger(__name__)

DAILY_RESULTS_PATH = Path("backtests/sma_baseline_results.json")
HOURLY_RESULTS_PATH = Path("backtests/sma_hourly_results.json")


def load_json_results(path: Path) -> dict[str, dict[str, Any]]:
    """Load backtest metrics from either a ticker mapping or a result list."""
    if not path.exists():
        log.warning("Results file not found: %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return {
                ticker: item
                for ticker, item in data.items()
                if isinstance(item, dict) and "ticker" in item
            }
        if isinstance(data, list):
            return {item["ticker"]: item for item in data if "ticker" in item}
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        log.error("Failed to load results from %s: %s", path, exc)
    return {}


def main() -> None:
    log.info("Comparing Daily vs Hourly SMA Backtest Performance...")

    daily_results = load_json_results(DAILY_RESULTS_PATH)
    hourly_results = load_json_results(HOURLY_RESULTS_PATH)
    all_tickers = sorted(set(daily_results) | set(hourly_results))

    if not all_tickers:
        log.error("No backtest results available for comparison.")
        return

    print("\n" + "=" * 105)
    print(f"{'SIDE-BY-SIDE COMPARISON: DAILY SMA(50/200) vs HOURLY SMA(20/50)':^105}")
    print("=" * 105)
    header = (
        f"{'Ticker':<8} | {'Daily Ret%':<10} {'Hourly Ret%':<11} | "
        f"{'Daily Sharpe':<12} {'Hourly Sharpe':<13} | "
        f"{'Daily Trades':<12} {'Hourly Trades':<13}"
    )
    print(header)
    print("-" * 105)

    for ticker in all_tickers:
        daily = daily_results.get(ticker, {})
        hourly = hourly_results.get(ticker, {})
        daily_return = f"{daily.get('total_return_pct', 0.0):.2f}%" if daily else "N/A"
        hourly_return = f"{hourly.get('total_return_pct', 0.0):.2f}%" if hourly else "N/A"
        daily_sharpe = f"{daily.get('sharpe_ratio', 0.0):.2f}" if daily else "N/A"
        hourly_sharpe = f"{hourly.get('sharpe_ratio', 0.0):.2f}" if hourly else "N/A"
        daily_trades = str(daily.get("n_trades", "N/A")) if daily else "N/A"
        hourly_trades = str(hourly.get("n_trades", "N/A")) if hourly else "N/A"

        print(
            f"{ticker:<8} | {daily_return:<10} {hourly_return:<11} | "
            f"{daily_sharpe:<12} {hourly_sharpe:<13} | "
            f"{daily_trades:<12} {hourly_trades:<13}"
        )

    print("=" * 105)


if __name__ == "__main__":
    main()
