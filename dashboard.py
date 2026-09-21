"""
dashboard.py — Streamlit + Plotly UI for StockOracle. Read-mostly: it shows
what's already in the database (prices, indicators, sentiment, predictions,
signals, backtest results) and offers a button to run the pipeline, but it
never places a trade — it can't, since nothing in this project does.

Run with:
    streamlit run dashboard.py

This file is NOT meant to be run with plain `python dashboard.py` — Streamlit
apps need the `streamlit run` launcher to work (it's what sets up the web
server and the script-reruns-on-interaction model). If someone does run it
directly, the import below fails fast with a clear message instead of an
unrelated traceback.
"""

from __future__ import annotations

import sys

import config
import database

log = config.get_logger(__name__)

try:
    import streamlit as st
except ImportError:
    print(
        "Streamlit is not installed (`pip install streamlit`), or this file was run with "
        "`python dashboard.py` instead of `streamlit run dashboard.py`. Either way, this "
        "dashboard can't start.",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    st.error("Plotly is not installed. Run `pip install plotly` and restart the dashboard.")
    st.stop()


st.set_page_config(page_title="StockOracle", layout="wide")

database.init_db()


def _dry_run_badge() -> None:
    if config.DRY_RUN:
        st.success("DRY_RUN = True — research mode. No trades are being placed by this system.")
    else:
        st.error(
            "DRY_RUN = False — this instance is configured for live behavior. "
            "This should only be true after backtesting has validated the strategy."
        )


def _price_chart(conn, ticker: str):
    rows = database.get_price_and_indicators(conn, ticker)
    if not rows:
        st.info(f"No price history stored for {ticker} yet. Run the pipeline first.")
        return

    dates = [r["date"] for r in rows]
    closes = [r["close"] for r in rows]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25],
        vertical_spacing=0.03,
        subplot_titles=("Price + Bollinger Bands + SMAs", "RSI(14)", "MACD"),
    )

    fig.add_trace(go.Scatter(x=dates, y=closes, name="Close", line=dict(color="#1f77b4")), row=1, col=1)

    bb_upper = [r["bb_upper"] for r in rows]
    bb_lower = [r["bb_lower"] for r in rows]
    if any(v is not None for v in bb_upper):
        fig.add_trace(go.Scatter(x=dates, y=bb_upper, name="BB Upper", line=dict(color="gray", dash="dot"), opacity=0.6), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=bb_lower, name="BB Lower", line=dict(color="gray", dash="dot"), opacity=0.6), row=1, col=1)

    sma_50 = [r["sma_50"] for r in rows]
    sma_200 = [r["sma_200"] for r in rows]
    if any(v is not None for v in sma_50):
        fig.add_trace(go.Scatter(x=dates, y=sma_50, name="SMA(50)", line=dict(color="orange")), row=1, col=1)
    if any(v is not None for v in sma_200):
        fig.add_trace(go.Scatter(x=dates, y=sma_200, name="SMA(200)", line=dict(color="purple")), row=1, col=1)

    rsi = [r["rsi_14"] for r in rows]
    if any(v is not None for v in rsi):
        fig.add_trace(go.Scatter(x=dates, y=rsi, name="RSI(14)", line=dict(color="green")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.4, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="red", opacity=0.4, row=2, col=1)

    macd = [r["macd"] for r in rows]
    macd_signal = [r["macd_signal"] for r in rows]
    if any(v is not None for v in macd):
        fig.add_trace(go.Scatter(x=dates, y=macd, name="MACD", line=dict(color="#1f77b4")), row=3, col=1)
        fig.add_trace(go.Scatter(x=dates, y=macd_signal, name="Signal", line=dict(color="orange")), row=3, col=1)

    fig.update_layout(height=750, showlegend=True, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _signal_panel(conn, ticker: str) -> None:
    row = database.get_latest_signal(conn, ticker)
    if row is None:
        st.info("No signal generated yet for this ticker. Run the pipeline first.")
        return

    signal_color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}.get(row["signal"], "gray")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"### :{signal_color}[{row['signal']}]")
        st.metric("Confidence", f"{row['confidence']:.0%}")
        st.caption(f"As of {row['date']}" + (" (DRY_RUN)" if row["dry_run"] else " (LIVE)"))
    with col2:
        st.markdown("**Reasoning**")
        st.write(row["reasoning"] or "—")
        if row["thesis"]:
            st.markdown("**Analyst thesis**")
            st.write(row["thesis"])


def _sentiment_panel(conn, ticker: str) -> None:
    rows = database.get_all_sentiment(conn, ticker) if hasattr(database, "get_all_sentiment") else []
    if not rows:
        # Fallback for a database.py without get_all_sentiment (older Phase 2
        # copies) — same query, kept local so the dashboard degrades gracefully.
        rows = conn.execute(
            "SELECT * FROM sentiment_scores WHERE ticker = ? ORDER BY date ASC", (ticker,)
        ).fetchall()
    if not rows:
        st.info("No sentiment data stored yet for this ticker.")
        return

    by_source: dict[str, dict[str, list]] = {}
    for r in rows:
        by_source.setdefault(r["source_type"], {"dates": [], "scores": []})
        by_source[r["source_type"]]["dates"].append(r["date"])
        by_source[r["source_type"]]["scores"].append(r["score"])

    fig = go.Figure()
    for source, series in by_source.items():
        fig.add_trace(go.Scatter(x=series["dates"], y=series["scores"], name=source, mode="lines+markers"))
    fig.add_hline(y=0, line_color="gray", opacity=0.4)
    fig.update_layout(height=300, yaxis_title="Sentiment (-1 to 1)", margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _backtest_panel(conn, ticker: str) -> None:
    rows = database.get_backtest_results(conn, ticker) if hasattr(database, "get_backtest_results") else []
    if not rows:
        st.info("No backtest results stored yet. Run `python backtester.py` first.")
        return

    st.dataframe(
        [
            {
                "Run at": r["created_at"],
                "Model": r["model_name"],
                "Period": f"{r['start_date']} → {r['end_date']}",
                "Trades": r["num_trades"],
                "Win rate": f"{r['win_rate']:.0%}" if r["win_rate"] is not None else "—",
                "Strategy return": f"{r['strategy_return_pct']:+.1f}%",
                "Buy & hold return": f"{r['benchmark_return_pct']:+.1f}%",
                "Max drawdown": f"{r['max_drawdown_pct']:.1%}" if r["max_drawdown_pct"] is not None else "—",
                "Sharpe": f"{r['sharpe_ratio']:.2f}" if r["sharpe_ratio"] is not None else "—",
            }
            for r in rows
        ],
        use_container_width=True,
    )
    st.caption(
        "Backtested signals use only {prediction, technical} components — sentiment is excluded "
        "because it has no historical archive (see backtester.py's module docstring)."
    )


def main() -> None:
    st.title("📈 StockOracle")
    st.caption("Research tool — analysis and reasoning, not financial advice.")
    _dry_run_badge()

    with database.get_connection() as conn:
        tickers_in_db = [
            r["ticker"]
            for r in conn.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker").fetchall()
        ]
        watchlist_tickers = sorted(set(config.WATCHLIST) | set(tickers_in_db))

        st.sidebar.header("Watchlist")
        ticker = st.sidebar.selectbox("Ticker", watchlist_tickers or config.WATCHLIST)

        if st.sidebar.button("Run pipeline now", help="Fetch fresh data and regenerate signals for this ticker."):
            with st.spinner(f"Running pipeline for {ticker}..."):
                import main as pipeline_main
                pipeline_main.run_pipeline([ticker])
            st.sidebar.success("Pipeline run complete — refresh below to see updates.")

        st.sidebar.divider()
        st.sidebar.caption(f"Model: {config.MODEL_NAME if hasattr(config, 'MODEL_NAME') else 'predictor'}")
        st.sidebar.caption(f"Min confidence to act: {config.MIN_CONFIDENCE_TO_ACT:.0%}")

        st.header(f"{ticker}")
        _signal_panel(conn, ticker)

        st.subheader("Price & Technicals")
        _price_chart(conn, ticker)

        st.subheader("Sentiment History")
        _sentiment_panel(conn, ticker)

        st.subheader("Backtest Results")
        _backtest_panel(conn, ticker)


if __name__ == "__main__":
    main()
