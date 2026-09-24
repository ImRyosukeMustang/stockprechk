"""
backtester.py — walk-forward backtest of the predictor + decision_engine
strategy, with realistic execution mechanics and transaction costs.
 
This is the module most responsible for the project's "no look-ahead bias"
and "walk-forward validation, not random train/test split" hard rules, so
its design choices are documented in detail below rather than left implicit.
 
WALK-FORWARD DESIGN
--------------------
History is split into successive folds. For each fold:
  1. Train a fresh model on all data strictly BEFORE the fold's test window
     (an "expanding window" — each fold's training set is a superset of the
     previous fold's).
  2. Generate predictions/signals for the test window using ONLY that
     model and that window's own technical-indicator values (both computed
     trailing-only, never touching future bars — see technical.py).
  3. Simulate trading through the test window with realistic execution
     (next-bar entry, stop-loss/take-profit, transaction costs — see below).
  4. Advance to the next fold and repeat.
 
This means a "prediction" for any historical date was generated using only
a model trained on data that came before it — exactly reproducing what a
live system would have known at that point in time.
 
THE EMBARGO (why this is subtler than it looks)
-------------------------------------------------
The model's training LABEL for date `d` is "did price close higher
`horizon_days` days after `d`?" — which necessarily looks at date
`d + horizon_days`. If a fold's training set were allowed to include the
last `horizon_days` dates immediately before the test window starts, those
training rows' labels would depend on price action that falls INSIDE the
test window. The model would then be trained on information from the
future it's about to be "tested" on — a leakage bug that's easy to miss
because it only affects a few rows at each fold boundary, but it does mean
the walk-forward result would be silently optimistic.
 
The fix (standard in time-series ML, sometimes called a "purge" or
"embargo"): each fold's training set stops `horizon_days` bars before the
test window begins, not immediately before it. `_fold_boundaries()` below
enforces this — it is the single most important function in this file for
the "no look-ahead bias" hard rule, and any change to the fold logic should
preserve this property.
 
EXECUTION MECHANICS
--------------------
  - A BUY/SELL decision made using date `d`'s close and indicators is
    executed at date `d+1`'s OPEN (not `d`'s own close) — a signal can't be
    acted on before the bar that produced it has finished.
  - While in a position, `config.STOP_LOSS_PCT` / `config.TAKE_PROFIT_PCT`
    are checked against each day's own high/low. This is a standard
    daily-bar backtesting simplification: it assumes the stop/target price
    was reachable intraday, without modeling intraday path. It is optimistic
    on days where price gaps past both levels, and that's a known
    approximation, not a bug to "fix" without also modeling intraday data.
  - `config.TRANSACTION_COST_PCT` is deducted from BOTH the entry and exit
    price of every trade (round-trip cost, per the "include transaction
    costs" hard rule).
  - Long-only, one position at a time, sized as 100% of capital per trade
    (this backtester reports RETURNS, not portfolio-level position sizing —
    config.MAX_POSITION_PCT belongs to a portfolio-level allocator, which
    is out of scope here and noted as a limitation below).
 
OUT-OF-SAMPLE PREDICTION PERSISTENCE
--------------------------------------
Each fold's out-of-sample predictions (the same `probs` used to drive
trading decisions in `_simulate_trades`) are also persisted to the
`predictions` table, tagged with `model_name = f"{ticker}_h5_wf"` and the
originating `fold_id`. This is purely additive — it does not change any
trading/strategy logic — and exists so that downstream diagnostics (e.g.
conditional-edge-by-regime analysis) can query a full history of strictly
out-of-sample probabilities rather than only the 5 live-pipeline rows.
Because these are walk-forward OOS predictions (a different model per
fold), they carry the "_wf" suffix to distinguish them from the live
pipeline's `{ticker}_h5` model_name.
 
KNOWN LIMITATIONS (read before trusting a backtest result)
-------------------------------------------------------------
  - Sentiment is excluded from the backtested strategy. sentiment.py only
    ever has data for recent dates (see its module docstring), so a
    historical backtest cannot fairly evaluate a sentiment-driven strategy
    — there's no historical sentiment archive to test against. The
    backtested decision logic therefore only combines the {prediction,
    technical} components via decision_engine.combine_components(), not
    {prediction, sentiment, technical} as the live pipeline does. A live
    signal and a backtested signal for the "same" setup are NOT directly
    comparable for this reason.
  - Position sizing is all-or-nothing per trade; config.MAX_POSITION_PCT
    (portfolio-level risk limits) is not modeled here.
  - Stop-loss/take-profit intraday-fill assumption, noted above.
  - This backtests one ticker at a time. It does not model portfolio-level
    effects (correlation across positions, capital shared across tickers).
"""
 
from __future__ import annotations
 
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
 
import config
import database
import decision_engine
import predictor
 
log = config.get_logger(__name__)
 
REPORTS_DIR = Path(__file__).resolve().parent / "backtests"
 
# fold_size is the TEST WINDOW size (trading days) per fold, not the fold
# count — a bigger test window means fewer total folds for the same amount
# of history. Raised from 20 -> 60 to cut total fold count roughly 60 -> 20,
# trimming backtest runtime without touching model strength (n_estimators
# stays at 200 in predictor._get_model_backend / config).
DEFAULT_FOLD_SIZE = 60
 
 
def _fold_boundaries(n_rows: int, horizon_days: int, min_train_rows: int, fold_size: int) -> list[tuple[int, int, int]]:
    """
    Compute (train_end, test_start, test_end) index triples, expanding-window,
    with an embargo of `horizon_days` bars between training data and the test
    window to prevent label leakage across the fold boundary (see module
    docstring). All indices are positions into the full chronological feature
    DataFrame. `train_end` is exclusive (train on rows [0, train_end)).
    """
    folds = []
    test_start = min_train_rows + horizon_days
    while test_start < n_rows:
        train_end = test_start - horizon_days  # the embargo
        test_end = min(test_start + fold_size, n_rows)
        folds.append((train_end, test_start, test_end))
        test_start += fold_size
    return folds
 
 
def _prepare_dataset(conn: sqlite3.Connection, ticker: str, horizon_days: int):
    """Build the full feature+label DataFrame for `ticker`, same construction
    as predictor.train_model, but keeping the full frame (not dropping the
    unlabeled tail) since the backtester needs OHLC data through the very
    last row for trade execution even where the forward-label is unknown."""
    df = predictor.build_feature_frame(conn, ticker)
    if df is None or df.empty:
        return None
    df["future_close"] = df["close"].shift(-horizon_days)
    df["label"] = (df["future_close"] > df["close"]).astype("Int64")  # nullable int; NA for the unlabeled tail
    return df.reset_index(drop=True)
 
 
def _technical_component_from_row(row) -> tuple[float, str] | None:
    """Same RSI/MACD scoring as decision_engine._technical_component, but
    operating on an in-memory DataFrame row instead of a DB lookup, since the
    backtester is iterating an already-built feature frame."""
    sub_scores, descs = [], []
 
    if row["rsi_14"] is not None and not _is_nan(row["rsi_14"]):
        rsi = row["rsi_14"]
        rsi_score = max(-1.0, min(1.0, (50 - rsi) / 30))
        sub_scores.append(rsi_score)
        state = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
        descs.append(f"RSI {rsi:.1f} ({state})")
 
    if row["macd"] is not None and row["macd_signal"] is not None and not _is_nan(row["macd"]) and not _is_nan(row["macd_signal"]):
        macd_score = 1.0 if row["macd"] > row["macd_signal"] else -1.0 if row["macd"] < row["macd_signal"] else 0.0
        sub_scores.append(macd_score)
        state = "bullish crossover" if macd_score > 0 else "bearish crossover" if macd_score < 0 else "flat"
        descs.append(f"MACD {state}")
 
    if not sub_scores:
        return None
    return sum(sub_scores) / len(sub_scores), "; ".join(descs)
 
 
def _is_nan(x) -> bool:
    if x is None:
        return True
    try:
        return bool(x != x)  # NaN != NaN
    except Exception:
        return True
 
 
def run_walkforward_backtest(
    conn: sqlite3.Connection,
    ticker: str,
    horizon_days: int = predictor.DEFAULT_HORIZON_DAYS,
    min_train_rows: int = predictor.MIN_TRAINING_ROWS,
    fold_size: int = DEFAULT_FOLD_SIZE,
    feature_columns: list[str] | None = None,
) -> dict | None:
    """
    Run a full walk-forward backtest for `ticker`. Returns a metrics dict
    (also saved as JSON under backtests/), or None if there wasn't enough
    data or a required dependency is missing.
    """
    backend_name, factory = predictor._get_model_backend()
    feature_columns = list(feature_columns or predictor.FEATURE_COLUMNS)
    if factory is None:
        log.warning("No ML backend (xgboost or scikit-learn) available — cannot backtest %s.", ticker)
        return None
 
    df = _prepare_dataset(conn, ticker, horizon_days)
    if df is None:
        return None
 
    folds = _fold_boundaries(len(df), horizon_days, min_train_rows, fold_size)
    if not folds:
        log.warning(
            "Not enough history for %s to run even one walk-forward fold "
            "(have %d rows, need >= %d + horizon %d).",
            ticker, len(df), min_train_rows, horizon_days,
        )
        return None
 
    wf_model_name = f"{ticker}_h5_wf"
 
    # Collect an out-of-sample (signal, row) for every date covered by a fold.
    oos_predictions: dict[int, float] = {}  # row index -> probability_up
    feature_importance_totals = {column: 0.0 for column in feature_columns}
    feature_importance_folds = 0
    n_predictions_saved = 0
    for fold_id, (train_end, test_start, test_end) in enumerate(folds):
        train_slice = df.iloc[:train_end]
        train_slice = train_slice[train_slice["label"].notna()]
        if train_slice["label"].nunique() < 2:
            log.warning("%s: fold train slice [0:%d) has a single class — skipping fold.", ticker, train_end)
            continue
 
        X_train = predictor._sanitize_features(train_slice[feature_columns], expected_columns=feature_columns)
        y_train = train_slice["label"].astype(int)
 
        model = factory()
        model.fit(X_train, y_train)
        importances = getattr(model, "feature_importances_", None)
        if importances is not None and len(importances) == len(feature_columns):
            for column, importance in zip(feature_columns, importances):
                feature_importance_totals[column] += float(importance)
            feature_importance_folds += 1
 
        test_slice = df.iloc[test_start:test_end]
        X_test = predictor._sanitize_features(test_slice[feature_columns], expected_columns=feature_columns)
        probs = model.predict_proba(X_test)[:, 1]
        for idx, prob in zip(test_slice.index, probs):
            oos_predictions[idx] = float(prob)
 
        # Persist this fold's out-of-sample predictions to the `predictions`
        # table. Purely additive — does not affect trade simulation below,
        # which continues to read from the in-memory `oos_predictions` dict.
        for idx, prob in zip(test_slice.index, probs):
            pred_date = df.iloc[idx]["date"]
            date_str = pred_date.strftime("%Y-%m-%d") if hasattr(pred_date, "strftime") else str(pred_date)
            try:
                database.insert_prediction(
                    conn,
                    ticker,
                    date_str,
                    wf_model_name,
                    float(prob),
                    fold_id=fold_id,
                )
                n_predictions_saved += 1
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "%s: failed to save walk-forward prediction for date %s (fold %d): %s",
                    ticker, date_str, fold_id, exc,
                )
 
    if not oos_predictions:
        log.warning("%s: no out-of-sample predictions were produced — cannot backtest.", ticker)
        return None
 
    log.info(
        "%s: saved %d walk-forward out-of-sample predictions to predictions table (model_name=%s).",
        ticker, n_predictions_saved, wf_model_name,
    )
 
    trades, equity_curve = _simulate_trades(df, oos_predictions)
    metrics = _compute_metrics(df, oos_predictions, trades, equity_curve)
    if feature_importance_folds:
        metrics["feature_importances"] = {
            column: value / feature_importance_folds
            for column, value in feature_importance_totals.items()
        }
        log.info(
            "%s feature importance: %s",
            ticker,
            ", ".join(
                f"{column}={importance:.4f}"
                for column, importance in sorted(
                    metrics["feature_importances"].items(), key=lambda item: item[1], reverse=True
                )[:10]
            ),
        )
    else:
        metrics["feature_importances"] = {}
    labeled_oos = [idx for idx in oos_predictions if df.iloc[idx]["label"] is not None and not _is_nan(df.iloc[idx]["label"])]
    if labeled_oos:
        metrics["holdout_accuracy"] = sum(
            int((oos_predictions[idx] >= 0.5) == bool(df.iloc[idx]["label"]))
            for idx in labeled_oos
        ) / len(labeled_oos)
    else:
        metrics["holdout_accuracy"] = None
    metrics.update(
        {
            "ticker": ticker,
            "horizon_days": horizon_days,
            "backend": backend_name,
            "n_folds": len(folds),
            "n_oos_days": len(oos_predictions),
            "run_at": datetime.now(timezone.utc).isoformat(),
            "feature_columns": feature_columns,
        }
    )
 
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{ticker.upper()}_h{horizon_days}_backtest.json"
    with open(report_path, "w") as f:
        json.dump({**metrics, "trades": trades}, f, indent=2, default=str)
    log.info("Backtest report for %s written to %s", ticker, report_path)
 
    # Also persist a queryable summary row to SQLite (backtest_results), so
    # dashboard.py (Phase 4) can list/compare backtest runs without needing
    # to read JSON files off disk. The JSON report remains the source of
    # truth for the full trade log / equity curve.
    oos_dates = [df.iloc[idx]["date"].strftime("%Y-%m-%d") for idx in sorted(oos_predictions.keys())]
    database.insert_backtest_result(
        conn,
        {
            "ticker": ticker,
            "model_name": backend_name,
            "start_date": oos_dates[0] if oos_dates else "",
            "end_date": oos_dates[-1] if oos_dates else "",
            "train_window_days": min_train_rows,
            "test_window_days": fold_size,
            "transaction_cost_pct": config.TRANSACTION_COST_PCT,
            "num_trades": metrics["n_trades"],
            "win_rate": metrics["win_rate"],
            "strategy_return_pct": metrics["total_return_pct"],
            "benchmark_return_pct": metrics["buy_hold_return_pct"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "detail_json": json.dumps({"report_path": str(report_path)}),
        },
    )
 
    log.info(
        "%s backtest: %d trades, win_rate=%.1f%%, strategy_return=%.1f%%, buy_hold_return=%.1f%%, max_drawdown=%.1f%%",
        ticker, metrics["n_trades"], metrics["win_rate"] * 100,
        metrics["total_return_pct"], metrics["buy_hold_return_pct"], metrics["max_drawdown_pct"] * 100,
    )
    return metrics
 
 
def _simulate_trades(df, oos_predictions: dict[int, float]) -> tuple[list[dict], list[dict]]:
    """
    Walk chronologically through the out-of-sample rows, generating a signal
    each day from {prediction, technical} components and simulating
    execution. Returns (trades, equity_curve). Long-only, one position at a
    time, all-or-nothing sizing (see module docstring limitations).
    """
    oos_indices = sorted(oos_predictions.keys())
    trades: list[dict] = []
    equity_curve: list[dict] = []
 
    capital = 1.0  # normalized starting capital; results reported as % return
    position: dict | None = None
 
    for i, idx in enumerate(oos_indices):
        row = df.iloc[idx]
        prob_up = oos_predictions[idx]
        pred_score = (prob_up - 0.5) * 2
        pred_component = (pred_score, f"model P(up)={prob_up:.0%}")
        tech_component = _technical_component_from_row(row)
 
        components = {"prediction": pred_component}
        if tech_component is not None:
            components["technical"] = tech_component
        decision = decision_engine.combine_components(components)
 
        # Mark-to-market today's equity before acting on today's decision.
        if position is not None:
            unrealized = (row["close"] - position["entry_price"]) / position["entry_price"]
            equity_curve.append({"date": row["date"].strftime("%Y-%m-%d"), "equity": capital * (1 + unrealized)})
        else:
            equity_curve.append({"date": row["date"].strftime("%Y-%m-%d"), "equity": capital})
 
        has_next_bar = idx + 1 < len(df)
 
        # Manage an open position: stop-loss / take-profit against today's range first.
        if position is not None:
            stop_price = position["entry_price"] * (1 - config.STOP_LOSS_PCT)
            target_price = position["entry_price"] * (1 + config.TAKE_PROFIT_PCT)
            exit_price = None
            exit_reason = None
            if row["low"] <= stop_price:
                exit_price, exit_reason = stop_price, "stop_loss"
            elif row["high"] >= target_price:
                exit_price, exit_reason = target_price, "take_profit"
            elif decision["signal"] == "SELL" and has_next_bar:
                exit_price, exit_reason = df.iloc[idx + 1]["open"], "sell_signal"
 
            if exit_price is not None:
                exit_price *= (1 - config.TRANSACTION_COST_PCT)
                trade_return = (exit_price - position["entry_price"]) / position["entry_price"]
                capital *= (1 + trade_return)
                trades.append(
                    {
                        "entry_date": position["entry_date"],
                        "exit_date": row["date"].strftime("%Y-%m-%d"),
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "return_pct": trade_return * 100,
                        "exit_reason": exit_reason,
                    }
                )
                position = None
 
        # Consider a new entry only if flat and there's a next bar to enter on.
        if position is None and decision["signal"] == "BUY" and has_next_bar:
            entry_price = df.iloc[idx + 1]["open"] * (1 + config.TRANSACTION_COST_PCT)
            position = {"entry_price": entry_price, "entry_date": df.iloc[idx + 1]["date"].strftime("%Y-%m-%d")}
 
    # Force-close any position still open at the end of the OOS window, at
    # the last known close, so metrics reflect a fully realized backtest.
    if position is not None:
        last_row = df.iloc[oos_indices[-1]]
        exit_price = last_row["close"] * (1 - config.TRANSACTION_COST_PCT)
        trade_return = (exit_price - position["entry_price"]) / position["entry_price"]
        capital *= (1 + trade_return)
        trades.append(
            {
                "entry_date": position["entry_date"],
                "exit_date": last_row["date"].strftime("%Y-%m-%d"),
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "return_pct": trade_return * 100,
                "exit_reason": "end_of_backtest",
            }
        )
 
    return trades, equity_curve
 
 
def _compute_metrics(df, oos_predictions: dict[int, float], trades: list[dict], equity_curve: list[dict]) -> dict:
    n_trades = len(trades)
    wins = [t for t in trades if t["return_pct"] > 0]
    win_rate = len(wins) / n_trades if n_trades else 0.0
    avg_trade_return_pct = sum(t["return_pct"] for t in trades) / n_trades if n_trades else 0.0
 
    total_return_pct = 0.0
    if equity_curve:
        total_return_pct = (equity_curve[-1]["equity"] - 1.0) * 100
 
    # Buy-and-hold benchmark over the same OOS window, for comparison.
    oos_indices = sorted(oos_predictions.keys())
    buy_hold_return_pct = 0.0
    if oos_indices:
        start_price = df.iloc[oos_indices[0]]["close"]
        end_price = df.iloc[oos_indices[-1]]["close"]
        buy_hold_return_pct = (end_price - start_price) / start_price * 100
 
    # Max drawdown from the equity curve.
    peak = -float("inf")
    max_drawdown = 0.0
    for point in equity_curve:
        peak = max(peak, point["equity"])
        drawdown = (point["equity"] - peak) / peak if peak > 0 else 0.0
        max_drawdown = min(max_drawdown, drawdown)
 
    # Rough annualized Sharpe-like ratio from daily equity returns (0% risk-free rate assumed).
    sharpe_ratio = None
    if len(equity_curve) > 2:
        daily_returns = []
        for prev, curr in zip(equity_curve, equity_curve[1:]):
            if prev["equity"] > 0:
                daily_returns.append((curr["equity"] - prev["equity"]) / prev["equity"])
        if daily_returns:
            mean_r = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_r) ** 2 for r in daily_returns) / len(daily_returns)
            std_r = variance ** 0.5
            if std_r > 0:
                sharpe_ratio = (mean_r / std_r) * (252 ** 0.5)
 
    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_trade_return_pct": avg_trade_return_pct,
        "total_return_pct": total_return_pct,
        "buy_hold_return_pct": buy_hold_return_pct,
        "max_drawdown_pct": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
    }
 
 
def backtest_watchlist(conn: sqlite3.Connection, tickers: list[str] | None = None, horizon_days: int = predictor.DEFAULT_HORIZON_DAYS) -> dict[str, dict | None]:
    tickers = tickers or config.WATCHLIST
    results: dict[str, dict | None] = {}
    for ticker in tickers:
        try:
            results[ticker] = run_walkforward_backtest(conn, ticker, horizon_days)
        except Exception as exc:
            log.error("Unexpected error backtesting %s: %s", ticker, exc)
            results[ticker] = None
    return results
 
 
def run_vol_targeted_backtest(conn: sqlite3.Connection, ticker: str) -> dict | None:
    """Backtest the SMA(50/200) plus volatility-targeting strategy."""
    import pandas as pd
    import vol_target

    rows = database.get_price_history(conn, ticker)
    if not rows or len(rows) < 200:
        log.warning("Not enough history for SMA+vol backtest on %s.", ticker)
        return None

    df = pd.DataFrame([dict(row) for row in rows])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["sma50"] = df["close"].rolling(window=50, min_periods=50).mean()
    df["sma200"] = df["close"].rolling(window=200, min_periods=200).mean()
    tradeable = df[df["sma200"].notna()].reset_index(drop=True)
    if len(tradeable) < 2:
        log.warning("Not enough tradeable bars for %s (need >= 2 after SMA200).", ticker)
        return None

    cost = config.TRANSACTION_COST_PCT
    target_vol = config.TARGET_VOL
    lookback = config.VOL_LOOKBACK_DAYS
    min_rebalance = config.MIN_REBALANCE_DELTA
    capital = 1.0
    current_position = 0.0
    entry_price = None
    entry_date = None
    trades: list[dict] = []
    equity_curve: list[dict] = []
    n_rebalances = 0
    invested_days = 0

    for i, row in tradeable.iterrows():
        if current_position > 0 and entry_price is not None:
            unrealized = current_position * (row["close"] - entry_price) / entry_price
            equity = capital * (1 + unrealized)
        else:
            equity = capital
        equity_curve.append({"date": row["date"].strftime("%Y-%m-%d"), "equity": equity})
        if current_position > 0:
            invested_days += 1

        if row["sma50"] > row["sma200"]:
            closes_up_to_now = tradeable["close"].iloc[: i + 1]
            target_position = vol_target.compute_position_size(
                closes_up_to_now, target_vol=target_vol, lookback=lookback
            )
        else:
            target_position = 0.0

        if abs(target_position - current_position) < min_rebalance or i + 1 >= len(tradeable):
            continue

        next_bar = tradeable.iloc[i + 1]
        next_open = next_bar["open"]
        next_date = next_bar["date"].strftime("%Y-%m-%d")
        if current_position > 0 and entry_price is not None:
            exit_price = next_open * (1 - cost)
            realized = current_position * (exit_price - entry_price) / entry_price
            capital *= 1 + realized
            trades.append({
                "entry_date": entry_date,
                "exit_date": next_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "position_size": current_position,
                "return_pct": realized * 100,
            })

        if target_position > 0:
            entry_price = next_open * (1 + cost)
            entry_date = next_date
            current_position = target_position
        else:
            entry_price = None
            entry_date = None
            current_position = 0.0
        n_rebalances += 1

    if current_position > 0 and entry_price is not None:
        last_row = tradeable.iloc[-1]
        exit_price = last_row["close"] * (1 - cost)
        realized = current_position * (exit_price - entry_price) / entry_price
        capital *= 1 + realized
        trades.append({
            "entry_date": entry_date,
            "exit_date": last_row["date"].strftime("%Y-%m-%d"),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_size": current_position,
            "return_pct": realized * 100,
        })

    n_trades = len(trades)
    win_rate = sum(trade["return_pct"] > 0 for trade in trades) / n_trades if n_trades else 0.0
    total_return_pct = (capital - 1.0) * 100
    first_close = tradeable.iloc[0]["close"]
    last_close = tradeable.iloc[-1]["close"]
    buy_hold_return_pct = (last_close - first_close) / first_close * 100

    peak = -float("inf")
    max_dd = 0.0
    for point in equity_curve:
        peak = max(peak, point["equity"])
        drawdown = (point["equity"] - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, drawdown)

    sharpe = None
    if len(equity_curve) > 2:
        daily_returns = [
            (current["equity"] - previous["equity"]) / previous["equity"]
            for previous, current in zip(equity_curve, equity_curve[1:])
            if previous["equity"] > 0
        ]
        if daily_returns:
            mean_return = sum(daily_returns) / len(daily_returns)
            variance = sum((value - mean_return) ** 2 for value in daily_returns) / len(daily_returns)
            std_return = variance ** 0.5
            if std_return > 0:
                sharpe = mean_return / std_return * (252 ** 0.5)

    exposure_pct = invested_days / len(tradeable) * 100 if len(tradeable) else 0.0
    metrics = {
        "ticker": ticker,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "total_return_pct": total_return_pct,
        "buy_hold_return_pct": buy_hold_return_pct,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "exposure_pct": exposure_pct,
        "n_rebalances": n_rebalances,
    }

    database.insert_backtest_result(conn, {
        "ticker": ticker,
        "model_name": f"{ticker}_sma_vol",
        "start_date": tradeable.iloc[0]["date"].strftime("%Y-%m-%d"),
        "end_date": tradeable.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "train_window_days": 0,
        "test_window_days": len(tradeable),
        "transaction_cost_pct": cost,
        "num_trades": n_trades,
        "win_rate": win_rate,
        "strategy_return_pct": total_return_pct,
        "benchmark_return_pct": buy_hold_return_pct,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "detail_json": json.dumps({"exposure_pct": exposure_pct, "n_rebalances": n_rebalances}),
    })
    log.info(
        "%s SMA+Vol backtest: %d trades, %d rebalances, win_rate=%.1f%%, return=%.1f%%, B&H=%.1f%%, max_dd=%.1f%%, exposure=%.1f%%",
        ticker, n_trades, n_rebalances, win_rate * 100, total_return_pct,
        buy_hold_return_pct, max_dd * 100, exposure_pct,
    )
    return metrics


def backtest_vol_targeted_watchlist(
    conn: sqlite3.Connection, tickers: list[str] | None = None
) -> dict[str, dict | None]:
    """Run the volatility-targeted backtest for every ticker."""
    tickers = tickers or config.WATCHLIST
    results: dict[str, dict | None] = {}
    for ticker in tickers:
        try:
            results[ticker] = run_vol_targeted_backtest(conn, ticker)
        except Exception as exc:
            log.error("Unexpected error in SMA+vol backtest for %s: %s", ticker, exc)
            results[ticker] = None
    return results


if __name__ == "__main__":
    database.init_db()
    with database.get_connection() as conn:
        log.info("=== Running ML walk-forward backtest ===")
        ml_summary = backtest_watchlist(conn)
        log.info("=== Running SMA + Volatility-Targeted backtest ===")
        vt_summary = backtest_vol_targeted_watchlist(conn)
        log.info("=== ML Backtest summary ===")
        for ticker, metrics in ml_summary.items():
            log.info("%s: %s", ticker, metrics)
        log.info("=== SMA+Vol Backtest summary ===")
        for ticker, metrics in vt_summary.items():
            log.info("%s: %s", ticker, metrics)
