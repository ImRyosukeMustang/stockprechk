"""
dashboard.py — Streamlit + Plotly UI for StockOracle.
"""

from __future__ import annotations

import sys

import config
import database
import alpaca_account
import pattern_analysis
import portfolio
import risk_controls

log = config.get_logger(__name__)

try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed.", file=sys.stderr)
    sys.exit(1)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    st.error("Plotly is not installed. Run `pip install plotly`.")
    st.stop()

try:
    import backtester
except ImportError:
    backtester = None

st.set_page_config(page_title="StockOracle", layout="wide")
database.init_db()


def _dry_run_badge() -> None:
    if config.DRY_RUN:
        st.success("DRY_RUN = True — research mode. No trades are being placed by this system.")
    else:
        st.error("DRY_RUN = False — this instance is configured for live behavior.")


def _system_health_panel(conn) -> None:
    st.subheader("System Health")
    col1, col2, col3 = st.columns(3)

    try:
        conn.execute("SELECT 1").fetchone()
        col1.success("✅ Database reachable")
    except Exception as e:
        col1.error(f"❌ Database error: {e}")

    try:
        account = alpaca_account.get_paper_account()
        if account and isinstance(account, dict):
            equity = account.get('equity', 0)
            col2.success(f"✅ Alpaca connected (${equity:,.0f})")
        else:
            col2.error("❌ Alpaca unavailable")
    except Exception as exc:
        col2.error(f"❌ Alpaca unavailable: {exc}")

    try:
        row = conn.execute("SELECT started_at, status FROM pipeline_runs ORDER BY id DESC LIMIT 1").fetchone()
        if row and row["started_at"]:
            date_str = str(row["started_at"])[:10]
            col3.info(f"⏱️ Last run: {date_str} ({row['status']})")
        else:
            col3.warning("⚠️ No pipeline runs yet")
    except Exception as exc:
        col3.warning(f"⚠️ Pipeline status unavailable: {exc}")

    try:
        fresh = conn.execute("SELECT COUNT(*) FROM data_health WHERE last_price_fetch >= datetime('now', '-1 day')").fetchone()[0]
        stale = conn.execute("SELECT COUNT(*) FROM data_health WHERE last_price_fetch < datetime('now', '-3 day')").fetchone()[0]
        col4, col5 = st.columns(2)
        col4.metric("Fresh Data (<24h)", fresh)
        col5.metric("Stale Data (>3d)", stale)
    except Exception as exc:
        st.caption(f"Data health unavailable: {exc}")


def _price_chart(conn, ticker: str):
    rows = database.get_price_and_indicators(conn, ticker)
    if not rows:
        st.info(f"No price history stored for {ticker} yet.")
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
    row = conn.execute(
        """
        SELECT * FROM signals
        WHERE ticker = ? AND reasoning LIKE 'SMA %'
        ORDER BY date DESC, created_at DESC, id DESC LIMIT 1
        """,
        (ticker,),
    ).fetchone()
    if row is None:
        st.info("No SMA signal generated yet for this ticker.")
        return

    signal_color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}.get(row["signal"], "gray")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"### :{signal_color}[{row['signal']}]")
        conf = row["confidence"]
        st.metric("Confidence", f"{conf:.0%}" if conf is not None else "N/A")
        dry_run_flag = row["dry_run"] if "dry_run" in row.keys() else 1
        st.caption(f"As of {row['date']}" + (" (DRY_RUN)" if dry_run_flag else " (LIVE)"))
    with col2:
        st.markdown("**Strategy: SMA 50/200**")
        st.markdown("**Reasoning**")
        st.write(row["reasoning"] or "—")
        if row["thesis"]:
            st.markdown("**Analyst thesis**")
            st.write(row["thesis"])


def _sentiment_panel(conn, ticker: str) -> None:
    rows = database.get_all_sentiment(conn, ticker) if hasattr(database, "get_all_sentiment") else []
    if not rows:
        rows = conn.execute("SELECT * FROM sentiment_scores WHERE ticker = ? ORDER BY date ASC", (ticker,)).fetchall()
    if not rows:
        st.info("No sentiment data stored yet.")
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
        st.info("No backtest results stored yet.")
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


def _latest_sma_data(conn, tickers: list[str]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for ticker in tickers:
        row = conn.execute(
            """
            SELECT p.date, t.sma_50, t.sma_200
            FROM prices p
            LEFT JOIN technical_indicators t ON t.ticker = p.ticker AND t.date = p.date
            WHERE p.ticker = ? ORDER BY p.date DESC LIMIT 1
            """,
            (ticker,),
        ).fetchone()
        if row is None:
            latest[ticker] = {"date": None, "regime": "unknown", "sma50": None, "sma200": None, "position_size": 0.0}
            continue
        sma50 = row["sma_50"]
        sma200 = row["sma_200"]
        regime = "unknown" if sma50 is None or sma200 is None else "bullish" if sma50 > sma200 else "bearish"
        latest[ticker] = {"date": row["date"], "regime": regime, "sma50": sma50, "sma200": sma200, "position_size": 1.0 if regime == "bullish" else 0.0}
    return latest


def _sma_regime_panel(conn, tickers: list[str]) -> dict[str, dict]:
    data = _latest_sma_data(conn, tickers)
    st.subheader("SMA Regime")
    st.dataframe(
        [
            {
                "Ticker": ticker,
                "Regime": values["regime"],
                "SMA50": f"{values['sma50']:.2f}" if values["sma50"] is not None else "N/A",
                "SMA200": f"{values['sma200']:.2f}" if values["sma200"] is not None else "N/A",
                "As of": values["date"] or "N/A",
            }
            for ticker, values in data.items()
        ],
        use_container_width=True,
    )
    return data


def _portfolio_allocation_panel(conn, sma_data: dict[str, dict]) -> None:
    allocation = portfolio.compute_portfolio_allocation(conn, sma_data)
    summary = portfolio.compute_portfolio_summary(allocation)

    st.subheader("Portfolio Allocation")
    st.dataframe(
        [{"Ticker": ticker, "Weight": f"{weight:.1%}"} for ticker, weight in allocation.items()],
        use_container_width=True,
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Total exposure", f"{summary['total_exposure']:.1%}")
    col2.metric("Cash", f"{summary['cash_pct']:.1%}")
    col3.metric("Positions", summary["n_positions"])


def _pattern_memory_panel(conn, ticker: str) -> None:
    try:
        result = pattern_analysis.compute_probability(conn, ticker)
    except Exception:
        result = None

    st.subheader("Pattern Memory")
    if not result or not result.get("pattern"):
        st.info("No pattern history is available for this ticker.")
        return

    pattern = result["pattern"]
    stats = result.get("pattern_stats", {})

    left, right = st.columns(2)
    with left:
        st.write(", ".join(f"{key}: {value}" for key, value in pattern.items() if key != "date"))
        st.metric("Probability up", f"{result.get('probability_up', 0):.0%}")
        st.metric("Confidence", f"{result.get('confidence', 0):.0%}")
    with right:
        st.write(f"This pattern has appeared {stats.get('count', 0)} times on {ticker}. Hit rate: {stats.get('win_rate', 0):.0%}.")
        st.write(f"Average 5-day return: {stats.get('avg_5d', 0):+.2f}%")
        st.write(f"Average 20-day return: {stats.get('avg_20d', 0):+.2f}%")

    st.write(result.get("reasoning", ""))
    if result.get("news_events"):
        st.dataframe(
            [{"Date": e["date"], "Headline": e["headline"], "Event": e["event_type"] or "other", "Sentiment": e["sentiment"], "Summary": e["summary"] or ""} for e in result["news_events"]],
            use_container_width=True,
        )


def _paper_portfolio_panel() -> None:
    st.subheader("Paper Portfolio")
    account = alpaca_account.get_paper_account()
    if account is None:
        st.info("Alpaca paper-account data is unavailable.")
        return
    if not isinstance(account, dict):
        st.error("Unexpected account format from Alpaca.")
        return

    columns = st.columns(4)
    columns[0].metric("Equity", f"${account.get('equity', 0):,.2f}")
    columns[1].metric("Cash", f"${account.get('cash', 0):,.2f}")
    columns[2].metric("Buying power", f"${account.get('buying_power', 0):,.2f}")
    columns[3].metric("Status", account.get("status", "unknown"))

    positions = alpaca_account.get_current_positions()
    if positions:
        st.dataframe(positions, use_container_width=True)
    else:
        st.caption("No open paper positions.")


def _data_health_panel(conn) -> None:
    st.subheader("Data Health")
    try:
        total = conn.execute("SELECT COUNT(DISTINCT ticker) FROM data_health").fetchone()[0]
        fresh = conn.execute("SELECT COUNT(*) FROM data_health WHERE last_price_fetch >= datetime('now', '-1 day')").fetchone()[0]
        stale = conn.execute("SELECT COUNT(*) FROM data_health WHERE last_price_fetch < datetime('now', '-3 day')").fetchone()[0]
        columns = st.columns(3)
        columns[0].metric("Tracked tickers", total)
        columns[1].metric("Fresh < 24h", fresh)
        columns[2].metric("Stale > 3d", stale)
    except Exception as exc:
        st.warning(f"Data health unavailable: {exc}")


def _risk_status_panel(conn) -> None:
    st.subheader("Risk Status")
    try:
        history = conn.execute("SELECT * FROM portfolio_history ORDER BY date DESC, id DESC LIMIT 1").fetchone()
        positions = conn.execute("SELECT ticker, entry_date, entry_price, shares FROM open_positions ORDER BY ticker").fetchall()
    except Exception:
        st.info("Risk tables not found yet.")
        return

    if not history:
        st.info("No portfolio history yet.")
    else:
        current_prices = {}
        for row in positions:
            price = conn.execute("SELECT close FROM prices WHERE ticker = ? ORDER BY date DESC LIMIT 1", (row["ticker"],)).fetchone()
            if price is not None:
                current_prices[row["ticker"]] = float(price["close"])
                
        current_value = risk_controls.get_portfolio_value(conn, current_prices)
        peak_value = risk_controls.get_portfolio_peak(conn)
        
        if peak_value is not None and peak_value > 0:
            threshold = peak_value * (1.0 - config.PORTFOLIO_STOP_LOSS_PCT)
            columns = st.columns(3)
            columns[0].metric("Portfolio value", f"${current_value:,.2f}")
            columns[1].metric("Peak value", f"${peak_value:,.2f}")
            columns[2].metric("Stop-loss threshold", f"${threshold:,.2f}")


def main() -> None:
    st.title("📈 StockOracle")
    st.caption("Research tool — analysis and reasoning, not financial advice.")
    _dry_run_badge()

    with database.get_connection() as conn:
        try:
            tickers_in_db = [r["ticker"] for r in conn.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker").fetchall()]
        except Exception:
            tickers_in_db = []
        
        watchlist_tickers = sorted(set(config.WATCHLIST) | set(tickers_in_db))
        
        if not watchlist_tickers:
            st.error("No tickers configured. Add them to config.py.")
            return

        st.sidebar.header("Watchlist")
        ticker = st.sidebar.selectbox("Ticker", watchlist_tickers)

        _system_health_panel(conn)
        sma_data = _sma_regime_panel(conn, config.WATCHLIST)
        _portfolio_allocation_panel(conn, sma_data)
        _risk_status_panel(conn)
        _data_health_panel(conn)
        _paper_portfolio_panel()

        if st.sidebar.button("Run pipeline now"):
            with st.spinner(f"Running pipeline for {ticker}..."):
                try:
                    from main import run_pipeline
                    run_pipeline([ticker])
                    st.sidebar.success("Pipeline run complete.")
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(f"Pipeline failed: {exc}")

        if st.sidebar.button("Run backtest"):
            if backtester is None:
                st.sidebar.error("backtester not available.")
            else:
                with st.spinner(f"Running backtest for {ticker}..."):
                    try:
                        backtester.run_walkforward_backtest(conn, ticker)
                        st.sidebar.success("Backtest complete.")
                        st.rerun()
                    except Exception as exc:
                        st.sidebar.error(f"Backtest failed: {exc}")
      

        if st.sidebar.button("Test Alpaca Connection"):
            with st.spinner("Connecting to Alpaca..."):
                account = alpaca_account.get_paper_account()
                if account and isinstance(account, dict):
                    st.sidebar.success(f"✅ Connected! Equity: ${account.get('equity', 0):,.2f}")
                else:
                    st.sidebar.error("❌ Failed to connect. Check Streamlit Secrets.")

        st.sidebar.divider()
        st.sidebar.caption("Strategy: SMA 50/200")
        
        st.header(f"{ticker}")
        _signal_panel(conn, ticker)
        _pattern_memory_panel(conn, ticker)
        st.subheader("Price & Technicals")
        _price_chart(conn, ticker)
        st.subheader("Sentiment History")
        _sentiment_panel(conn, ticker)
        st.subheader("Backtest Results")
        _backtest_panel(conn, ticker)


if __name__ == "__main__":
    main()
