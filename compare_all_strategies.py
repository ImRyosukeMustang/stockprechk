"""Compare buy-and-hold, SMA, ML, and SMA plus volatility targeting."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import config

log = config.get_logger(__name__)
SMA_RESULTS_PATH = Path("backtests") / "sma_baseline_results.json"
DB_PATH = Path("stockoracle.db")


def load_sma_results() -> dict:
    if not SMA_RESULTS_PATH.exists():
        log.warning("%s not found. Run sma_baseline.py first.", SMA_RESULTS_PATH)
        return {}
    with open(SMA_RESULTS_PATH) as file:
        return json.load(file)


def load_backtest_rows() -> dict[str, dict[str, dict]]:
    """Return the latest ML and SMA+Vol row for each ticker."""
    if not DB_PATH.exists():
        log.warning("%s not found.", DB_PATH)
        return {}

    query = """
        SELECT ticker, model_name, strategy_return_pct, benchmark_return_pct,
               max_drawdown_pct, sharpe_ratio, num_trades, win_rate
        FROM backtest_results
        WHERE rowid IN (
            SELECT MAX(rowid) FROM backtest_results GROUP BY ticker, model_name
        )
    """
    results: dict[str, dict[str, dict]] = {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query).fetchall()
    except sqlite3.Error as exc:
        log.error("Failed to read backtest_results: %s", exc)
        return {}

    for row in rows:
        ticker = row["ticker"]
        model_name = row["model_name"] or ""
        results.setdefault(ticker, {})
        if model_name.endswith("_sma_vol"):
            results[ticker]["vol_target"] = dict(row)
        elif "xgboost" in model_name.lower() or model_name.endswith(("_h5_wf", "_h5")):
            results[ticker]["ml"] = dict(row)
    return results


def _fmt_pct(value, decimals: int = 1) -> str:
    try:
        return f"{float(value):+.{decimals}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_num(value, decimals: int = 2) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "n/a"


def _metric(row: dict, key: str, fallback=None):
    value = row.get(key, fallback) if row else fallback
    return value


def main() -> None:
    sma_results = load_sma_results()
    backtests = load_backtest_rows()
    header = (
        f"{'Ticker':<8}{'Strategy':<18}{'Return':>10}{'B&H':>10}"
        f"{'Sharpe':>9}{'MaxDD':>10}{'Trades':>8}"
    )
    print("\n" + "=" * len(header))
    print("ALL-STRATEGY COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    winners = {"buy_hold": 0, "sma": 0, "ml": 0, "vol_target": 0}
    metric_wins = {"return": {}, "sharpe": {}, "drawdown": {}}

    for ticker in config.WATCHLIST:
        sma = sma_results.get(ticker, {})
        ml = backtests.get(ticker, {}).get("ml", {})
        vt = backtests.get(ticker, {}).get("vol_target", {})
        rows = []
        if sma:
            rows.append(("sma", "SMA 50/200", sma, "total_return_pct", "buy_hold_return_pct", "n_trades"))
        if ml:
            rows.append(("ml", "ML (XGBoost)", ml, "strategy_return_pct", "benchmark_return_pct", "num_trades"))
        if vt:
            rows.append(("vol_target", "SMA+VolTarget", vt, "strategy_return_pct", "benchmark_return_pct", "num_trades"))

        buy_hold = None
        for _, _, row, _, benchmark_key, _ in rows:
            if row.get(benchmark_key) is not None:
                buy_hold = row[benchmark_key]
                break
        if buy_hold is None and sma:
            buy_hold = sma.get("buy_hold_return_pct")
        if buy_hold is not None:
            rows.insert(0, ("buy_hold", "Buy & Hold", {
                "return": buy_hold,
                "sharpe": None,
                "drawdown": None,
                "trades": 0,
            }, "return", "return", "trades"))

        for key, label, row, return_key, benchmark_key, trades_key in rows:
            if key == "buy_hold":
                return_value = row["return"]
                sharpe = None
                drawdown = None
                trades = 0
                benchmark = return_value
            else:
                return_value = row.get(return_key)
                sharpe = row.get("sharpe_ratio")
                drawdown = row.get("max_drawdown_pct")
                trades = row.get(trades_key, 0)
                benchmark = row.get(benchmark_key)
            print(
                f"{ticker:<8}{label:<18}{_fmt_pct(return_value):>10}"
                f"{_fmt_pct(benchmark):>10}{_fmt_num(sharpe):>9}"
                f"{_fmt_pct(drawdown):>10}{trades:>8}"
            )

        candidates = [(key, row) for key, _, row, _, _, _ in rows]
        for metric, values in (
            ("return", [(key, row.get("return", row.get("total_return_pct", row.get("strategy_return_pct")))) for key, row in candidates]),
            ("sharpe", [(key, row.get("sharpe_ratio")) for key, row in candidates]),
            ("drawdown", [(key, row.get("max_drawdown_pct")) for key, row in candidates]),
        ):
            valid = [(key, float(value)) for key, value in values if value is not None]
            if valid:
                best = min(valid, key=lambda item: item[1]) if metric == "drawdown" else max(valid, key=lambda item: item[1])
                metric_wins[metric][best[0]] = metric_wins[metric].get(best[0], 0) + 1
                if metric == "sharpe":
                    winners[best[0]] += 1
        print("-" * len(header))

    print("=" * len(header))
    print("WINNERS BY SHARPE (higher is better)")
    print("=" * len(header))
    for key, label in (("buy_hold", "Buy & Hold"), ("sma", "SMA 50/200 alone"), ("ml", "ML (XGBoost)"), ("vol_target", "SMA + Vol Targeting")):
        print(f"  {label:<24}{winners[key]} tickers")
    print("\nWINNERS BY METRIC")
    for metric in ("return", "sharpe", "drawdown"):
        print(f"  {metric:<10}{metric_wins[metric]}")

    recommendation = max(
        (key for key in ("sma", "ml", "vol_target") if metric_wins["sharpe"].get(key, 0) >= 0),
        key=lambda key: metric_wins["sharpe"].get(key, 0),
        default="vol_target",
    )
    labels = {"sma": "SMA 50/200", "ml": "ML (XGBoost)", "vol_target": "SMA + Vol Targeting"}
    print(f"\nRECOMMENDATION: {labels[recommendation]} leads the non-benchmark strategies by Sharpe wins.")


if __name__ == "__main__":
    main()
