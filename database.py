"""
database.py — SQLite schema and access helpers for StockOracle.
 
Design notes:
- Raw sqlite3 (no ORM), per the project's tech stack.
- One connection helper (`get_connection`) that every other function reuses;
  callers may also open their own connection with the same helper.
- `init_db()` is idempotent — safe to call on every startup. It only ever
  creates tables/indexes if they don't already exist; it never drops data.
- Every table has a `created_at` timestamp defaulting to UTC "now" at
  insert time, so later modules (backtester, dashboard) can reconstruct
  "what did we know and when" without look-ahead bias.
- Insert helpers use `INSERT OR IGNORE` / explicit uniqueness constraints
  where duplicates are meaningless (e.g. the same price bar fetched twice),
  so re-running data_fetcher.py is always safe.
"""
 
from __future__ import annotations
 
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence
 
import config
 
log = config.get_logger(__name__)
 
SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    date        TEXT    NOT NULL,          -- ISO date, e.g. '2026-09-19'
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      INTEGER NOT NULL,
    fetched_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, date)
);
 
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker, date);

CREATE TABLE IF NOT EXISTS data_health (
    ticker TEXT PRIMARY KEY,
    last_price_fetch TEXT,
    last_fundamentals_fetch TEXT,
    last_news_fetch TEXT,
    consecutive_failures INTEGER DEFAULT 0,
    last_error TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
 
CREATE TABLE IF NOT EXISTS news (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    source      TEXT    NOT NULL,          -- 'google_news', 'newsapi', etc.
    title       TEXT    NOT NULL,
    url         TEXT    NOT NULL,
    published_at TEXT,                     -- ISO datetime if the source gave one
    summary     TEXT,
    fetched_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, url)
);
 
CREATE INDEX IF NOT EXISTS idx_news_ticker ON news(ticker);

CREATE TABLE IF NOT EXISTS earnings_calendar (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    earnings_date TEXT NOT NULL,
    eps_estimate REAL,
    eps_actual REAL,
    revenue_estimate REAL,
    revenue_actual REAL,
    hour TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, earnings_date)
);

CREATE INDEX IF NOT EXISTS idx_earnings_calendar_ticker_date
    ON earnings_calendar(ticker, earnings_date);
 
CREATE TABLE IF NOT EXISTS reddit_posts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    subreddit   TEXT    NOT NULL,
    post_id     TEXT    NOT NULL,          -- reddit's own post id, for de-dup
    title       TEXT    NOT NULL,
    score       INTEGER,
    num_comments INTEGER,
    created_utc TEXT,                      -- ISO datetime of the reddit post
    url         TEXT,
    fetched_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(post_id)
);
 
CREATE INDEX IF NOT EXISTS idx_reddit_ticker ON reddit_posts(ticker);
 
CREATE TABLE IF NOT EXISTS fundamentals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    date        TEXT    NOT NULL,          -- date the snapshot was taken
    market_cap  REAL,
    pe_ratio    REAL,
    forward_pe  REAL,
    peg_ratio   REAL,
    eps         REAL,
    profit_margin REAL,
    revenue_growth REAL,
    earnings_growth REAL,
    debt_to_equity REAL,
    free_cashflow REAL,
    dividend_yield REAL,
    beta REAL,
    sector      TEXT,
    industry    TEXT,
    raw_json    TEXT,                      -- full yfinance .info blob, for anything not modeled above
    fetched_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, date)
);
 
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    source_type TEXT    NOT NULL,          -- 'news' | 'reddit' | 'llm'
    score       REAL    NOT NULL,          -- normalized -1..1
    label       TEXT,                      -- e.g. 'positive'/'neutral'/'negative'
    detail_json TEXT,                      -- model-specific extra info
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
 
CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date ON sentiment_scores(ticker, date);
 
CREATE TABLE IF NOT EXISTS technical_indicators (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    rsi_14      REAL,
    macd        REAL,
    macd_signal REAL,
    bb_upper    REAL,
    bb_lower    REAL,
    sma_50      REAL,
    sma_200     REAL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, date)
);
 
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    model_name  TEXT    NOT NULL,          -- e.g. 'xgboost_v1'
    probability_up REAL NOT NULL,          -- model's raw probability output
    fold_id     INTEGER,                   -- walk-forward fold index (NULL for live predictions)
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, date, model_name)
);
 
CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    signal      TEXT    NOT NULL,          -- 'BUY' | 'SELL' | 'HOLD'
    confidence  REAL    NOT NULL,          -- 0..1
    reasoning   TEXT,                      -- short structured reasoning string
    thesis      TEXT,                      -- full LLM-written thesis (Phase 4)
    dry_run     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
 
CREATE INDEX IF NOT EXISTS idx_signals_ticker_date ON signals(ticker, date);

CREATE TABLE IF NOT EXISTS patterns (
    pattern_id TEXT PRIMARY KEY,
    rsi_bucket TEXT NOT NULL,
    macd_state TEXT NOT NULL,
    bb_position TEXT NOT NULL,
    trend TEXT NOT NULL,
    volume TEXT NOT NULL,
    description TEXT,
    first_seen TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pattern_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    forward_5d_return REAL,
    forward_20d_return REAL,
    was_up_5d INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(pattern_id, ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_pattern_outcomes_pattern ON pattern_outcomes(pattern_id);
CREATE INDEX IF NOT EXISTS idx_pattern_outcomes_ticker_date ON pattern_outcomes(ticker, date);

CREATE TABLE IF NOT EXISTS news_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    headline TEXT NOT NULL,
    url TEXT,
    event_type TEXT,
    sentiment REAL,
    political INTEGER DEFAULT 0,
    category TEXT,
    summary TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, url)
);
 
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    status      TEXT    NOT NULL DEFAULT 'running',  -- 'running' | 'success' | 'failed'
    detail      TEXT
);
 
CREATE TABLE IF NOT EXISTS backtest_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT    NOT NULL,
    model_name          TEXT    NOT NULL,
    start_date          TEXT    NOT NULL,
    end_date            TEXT    NOT NULL,
    train_window_days   INTEGER NOT NULL,
    test_window_days    INTEGER NOT NULL,
    transaction_cost_pct REAL   NOT NULL,
    num_trades          INTEGER NOT NULL,
    win_rate            REAL,
    strategy_return_pct REAL    NOT NULL,
    benchmark_return_pct REAL   NOT NULL,     -- buy-and-hold over the same period
    max_drawdown_pct    REAL,
    sharpe_ratio        REAL,
    detail_json         TEXT,                 -- full equity curve + per-trade log
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);
 
CREATE INDEX IF NOT EXISTS idx_backtest_ticker ON backtest_results(ticker);

CREATE TABLE IF NOT EXISTS macro_indicators (
    date TEXT PRIMARY KEY,
    fed_funds_rate REAL,
    ten_year_yield REAL,
    cpi REAL,
    vix REAL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS prices_intraday (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    interval    TEXT    NOT NULL,          -- '1m', '5m', '1h', etc.
    timestamp   TEXT    NOT NULL,          -- ISO datetime (UTC)
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      INTEGER NOT NULL,
    vwap        REAL,
    trade_count INTEGER,
    fetched_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, interval, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_prices_intraday_ticker_interval
    ON prices_intraday(ticker, interval, timestamp);

CREATE TABLE IF NOT EXISTS sma_signal_journal (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    date                TEXT    NOT NULL,
    ticker              TEXT    NOT NULL,
    regime              TEXT    NOT NULL,       -- 'bullish' | 'bearish'
    sma50               REAL,
    sma200              REAL,
    close_at_signal     REAL,
    action              TEXT    NOT NULL,       -- 'BUY' | 'SELL' | 'HOLD'
    forward_5d_return   REAL,                   -- % return, filled in 5 days later
    forward_20d_return  REAL,                   -- % return, filled in 20 days later
    forward_60d_return  REAL,                   -- % return, filled in 60 days later
    was_correct_5d      INTEGER,                -- 1 if signal direction matched, 0 if not, NULL if N/A
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_journal_ticker_date ON sma_signal_journal(ticker, date);
CREATE INDEX IF NOT EXISTS idx_journal_date ON sma_signal_journal(date);
"""
 
 
@contextmanager
def get_connection(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """
    Yield a sqlite3 connection with sane defaults (row factory, foreign keys),
    committing on clean exit and rolling back on exception. Always closes.
 
    Usage:
        with get_connection() as conn:
            conn.execute(...)
    """
    path = str(db_path) if db_path is not None else str(config.DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
 
 
def init_db(db_path: Path | str | None = None) -> None:
    """Create all tables/indexes if they don't already exist. Safe to call repeatedly."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
        # Migration for DBs created before `fold_id` was added to `predictions`.
        # SCHEMA above already includes fold_id for brand-new databases (CREATE
        # TABLE IF NOT EXISTS is a no-op there), so this ALTER TABLE only ever
        # does real work against a pre-existing predictions table that lacks
        # the column. Idempotent: swallow "duplicate column name" and re-raise
        # anything else.
        try:
            conn.execute("ALTER TABLE predictions ADD COLUMN fold_id INTEGER")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise  # Re-raise real errors; ignore "already exists"
        for column in (
            "forward_pe", "peg_ratio", "profit_margin", "revenue_growth",
            "earnings_growth", "debt_to_equity", "free_cashflow", "beta",
        ):
            try:
                conn.execute(f"ALTER TABLE fundamentals ADD COLUMN {column} REAL")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
    log.info("Database initialized at %s", db_path or config.DB_PATH)
 
 
# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------
 
def insert_price_bar(
    conn: sqlite3.Connection,
    ticker: str,
    date: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO prices (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, date, open_, high, low, close, volume),
    )
 
 
def insert_price_bars(conn: sqlite3.Connection, ticker: str, rows: Sequence[dict[str, Any]]) -> int:
    """Bulk-insert price bars. Each row dict needs date/open/high/low/close/volume. Returns row count inserted."""
    conn.executemany(
        """
        INSERT OR IGNORE INTO prices (ticker, date, open, high, low, close, volume)
        VALUES (:ticker, :date, :open, :high, :low, :close, :volume)
        """,
        [{**row, "ticker": ticker} for row in rows],
    )
    return len(rows)
 
 
def insert_news_item(
    conn: sqlite3.Connection,
    ticker: str,
    source: str,
    title: str,
    url: str,
    published_at: str | None = None,
    summary: str | None = None,
) -> bool:
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO news (ticker, source, title, url, published_at, summary)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (ticker, source, title, url, published_at, summary),
    )
    return cursor.rowcount > 0


def insert_earnings_calendar_entry(
    conn: sqlite3.Connection,
    ticker: str,
    earnings_date: str,
    eps_estimate: float | None,
    eps_actual: float | None,
    revenue_estimate: float | None,
    revenue_actual: float | None,
    hour: str | None,
) -> bool:
    """Insert one earnings entry and return whether it was newly inserted."""
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO earnings_calendar
            (ticker, earnings_date, eps_estimate, eps_actual,
             revenue_estimate, revenue_actual, hour)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticker,
            earnings_date,
            eps_estimate,
            eps_actual,
            revenue_estimate,
            revenue_actual,
            hour,
        ),
    )
    return cursor.rowcount > 0
 
 
def insert_reddit_post(
    conn: sqlite3.Connection,
    ticker: str,
    subreddit: str,
    post_id: str,
    title: str,
    score: int | None,
    num_comments: int | None,
    created_utc: str | None,
    url: str | None,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO reddit_posts
            (ticker, subreddit, post_id, title, score, num_comments, created_utc, url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, subreddit, post_id, title, score, num_comments, created_utc, url),
    )
 
 
def insert_fundamentals(
    conn: sqlite3.Connection,
    ticker: str,
    date: str,
    market_cap: float | None,
    pe_ratio: float | None,
    eps: float | None,
    dividend_yield: float | None,
    sector: str | None,
    industry: str | None,
    raw_json: str | None,
    forward_pe: float | None = None,
    peg_ratio: float | None = None,
    profit_margin: float | None = None,
    revenue_growth: float | None = None,
    earnings_growth: float | None = None,
    debt_to_equity: float | None = None,
    free_cashflow: float | None = None,
    beta: float | None = None,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO fundamentals
            (ticker, date, market_cap, pe_ratio, forward_pe, peg_ratio, eps,
             profit_margin, revenue_growth, earnings_growth, debt_to_equity,
             free_cashflow, dividend_yield, beta, sector, industry, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, date, market_cap, pe_ratio, forward_pe, peg_ratio, eps,
         profit_margin, revenue_growth, earnings_growth, debt_to_equity,
         free_cashflow, dividend_yield, beta, sector, industry, raw_json),
    )
 
 
def replace_sentiment_score(
    conn: sqlite3.Connection,
    ticker: str,
    date: str,
    source_type: str,
    score: float,
    label: str | None,
    detail_json: str | None,
) -> None:
    """Idempotent write for a daily aggregate sentiment score. `sentiment_scores`
    has no UNIQUE constraint (it's designed to allow an append-only per-item log
    in the future), so callers writing one aggregate row per ticker/date/source
    must clear any prior aggregate for that key first — this does both steps
    in one call so re-running sentiment.py never accumulates duplicates."""
    conn.execute(
        "DELETE FROM sentiment_scores WHERE ticker = ? AND date = ? AND source_type = ?",
        (ticker, date, source_type),
    )
    conn.execute(
        """
        INSERT INTO sentiment_scores (ticker, date, source_type, score, label, detail_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (ticker, date, source_type, score, label, detail_json),
    )
 
 
def upsert_technical_indicators(
    conn: sqlite3.Connection,
    ticker: str,
    date: str,
    rsi_14: float | None,
    macd: float | None,
    macd_signal: float | None,
    bb_upper: float | None,
    bb_lower: float | None,
    sma_50: float | None,
    sma_200: float | None,
) -> None:
    """Insert or update the indicator row for (ticker, date). Uses the table's
    UNIQUE(ticker, date) constraint so recomputing indicators (e.g. after new
    price bars arrive) always reflects the latest calculation rather than
    appending stale duplicates."""
    conn.execute(
        """
        INSERT INTO technical_indicators
            (ticker, date, rsi_14, macd, macd_signal, bb_upper, bb_lower, sma_50, sma_200)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date) DO UPDATE SET
            rsi_14 = excluded.rsi_14,
            macd = excluded.macd,
            macd_signal = excluded.macd_signal,
            bb_upper = excluded.bb_upper,
            bb_lower = excluded.bb_lower,
            sma_50 = excluded.sma_50,
            sma_200 = excluded.sma_200
        """,
        (ticker, date, rsi_14, macd, macd_signal, bb_upper, bb_lower, sma_50, sma_200),
    )
 
 
def get_technical_indicators(conn: sqlite3.Connection, ticker: str, date: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM technical_indicators WHERE ticker = ? AND date = ?", (ticker, date)
    ).fetchone()
 
 
def get_latest_sentiment(conn: sqlite3.Connection, ticker: str, source_type: str | None = None) -> list[sqlite3.Row]:
    """Return the most recent sentiment_scores row(s) for a ticker, optionally
    filtered to one source_type ('news' | 'reddit' | 'llm')."""
    if source_type:
        return conn.execute(
            """
            SELECT * FROM sentiment_scores
            WHERE ticker = ? AND source_type = ?
            ORDER BY date DESC, created_at DESC LIMIT 1
            """,
            (ticker, source_type),
        ).fetchall()
    return conn.execute(
        """
        SELECT * FROM sentiment_scores
        WHERE ticker = ? AND date = (SELECT MAX(date) FROM sentiment_scores WHERE ticker = ?)
        """,
        (ticker, ticker),
    ).fetchall()
 
 
def get_price_and_indicators(conn: sqlite3.Connection, ticker: str) -> list[sqlite3.Row]:
    """Return one row per date with price OHLCV joined to that date's technical
    indicators (LEFT JOIN — early dates before enough history exist will have
    NULL indicator columns), ordered ascending by date. This is the base table
    predictor.py and backtester.py build feature matrices from."""
    return conn.execute(
        """
        SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, p.volume,
               t.rsi_14, t.macd, t.macd_signal, t.bb_upper, t.bb_lower, t.sma_50, t.sma_200
        FROM prices p
        LEFT JOIN technical_indicators t ON t.ticker = p.ticker AND t.date = p.date
        WHERE p.ticker = ?
        ORDER BY p.date ASC
        """,
        (ticker,),
    ).fetchall()
 
 
def get_all_sentiment(conn: sqlite3.Connection, ticker: str) -> list[sqlite3.Row]:
    """Return every stored sentiment_scores row for a ticker (all dates, all
    source types), ascending by date. predictor.py pivots this into
    per-date news/reddit/llm columns for the feature frame. As noted in
    sentiment.py, this will only have rows for the recent dates news/Reddit
    were actually fetched for — not a full historical archive."""
    return conn.execute(
        "SELECT * FROM sentiment_scores WHERE ticker = ? ORDER BY date ASC", (ticker,)
    ).fetchall()
 
 
def insert_prediction(
    conn: sqlite3.Connection,
    ticker: str,
    date: str,
    model_name: str,
    probability_up: float,
    fold_id: int | None = None,
) -> None:
    """Idempotent write for a model's prediction on (ticker, date, model_name).
    Relies on the table's UNIQUE(ticker, date, model_name) constraint, so
    re-running the predictor for a date it already scored updates that row
    in place rather than raising an IntegrityError or piling up duplicates.
    `fold_id` is the walk-forward fold index that produced this prediction
    (None for live-pipeline predictions, which aren't tied to a fold)."""
    conn.execute(
        """
        INSERT INTO predictions (ticker, date, model_name, probability_up, fold_id)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date, model_name) DO UPDATE SET
            probability_up = excluded.probability_up,
            fold_id = excluded.fold_id,
            created_at = datetime('now')
        """,
        (ticker, date, model_name, probability_up, fold_id),
    )
 
 
def get_latest_prediction(conn: sqlite3.Connection, ticker: str, model_name: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM predictions WHERE ticker = ? AND model_name = ?
        ORDER BY date DESC, created_at DESC LIMIT 1
        """,
        (ticker, model_name),
    ).fetchone()
 
 
def insert_signal(
    conn: sqlite3.Connection,
    ticker: str,
    date: str,
    signal: str,
    confidence: float,
    reasoning: str | None,
    thesis: str | None = None,
    dry_run: bool = True,
) -> int:
    """Append-only log of decision-engine output — one row per run, kept even
    across reruns for the same ticker/date, so signals.py's history shows how
    a call would have changed as new data came in over the course of a day.
    Returns the new row's id, so callers (llm_analyst.py) can attach a thesis
    to this exact row later via update_signal_thesis()."""
    cur = conn.execute(
        """
        INSERT INTO signals (ticker, date, signal, confidence, reasoning, thesis, dry_run)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, date, signal, confidence, reasoning, thesis, int(dry_run)),
    )
    return int(cur.lastrowid)
 
 
def get_latest_signal(conn: sqlite3.Connection, ticker: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM signals WHERE ticker = ? ORDER BY date DESC, created_at DESC LIMIT 1",
        (ticker,),
    ).fetchone()
 
 
def update_signal_thesis(conn: sqlite3.Connection, signal_id: int, thesis: str) -> None:
    """Attach an LLM-written thesis to an already-persisted signal row.
    llm_analyst.py calls this after decision_engine.py has already written
    the row — the thesis is a follow-up enrichment step, not part of the
    original signal decision, so it's a separate write rather than requiring
    every signal-writer to know about theses."""
    conn.execute("UPDATE signals SET thesis = ? WHERE id = ?", (thesis, signal_id))


def update_signal_reasoning(conn: sqlite3.Connection, signal_id: int, reasoning: str) -> None:
    """Append enrichment such as pattern-memory reasoning to a signal row."""
    conn.execute("UPDATE signals SET reasoning = ? WHERE id = ?", (reasoning, signal_id))


def insert_news_event(
    conn: sqlite3.Connection,
    ticker: str,
    date: str,
    headline: str,
    url: str | None,
    event_type: str | None,
    sentiment: float | None,
    political: int,
    category: str | None,
    summary: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO news_events
            (ticker, date, headline, url, event_type, sentiment, political, category, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, url) DO UPDATE SET
            date=excluded.date,
            headline=excluded.headline,
            event_type=excluded.event_type,
            sentiment=excluded.sentiment,
            political=excluded.political,
            category=excluded.category,
            summary=excluded.summary
        """,
        (ticker, date, headline, url, event_type, sentiment, political, category, summary),
    )
 
 
def insert_backtest_result(conn: sqlite3.Connection, result: dict[str, Any]) -> int:
    """Insert one backtest summary row. `result` keys must match the
    backtest_results columns (ticker, model_name, start_date, end_date,
    train_window_days, test_window_days, transaction_cost_pct, num_trades,
    win_rate, strategy_return_pct, benchmark_return_pct, max_drawdown_pct,
    sharpe_ratio, detail_json) — detail_json should already be a JSON string.
    Returns the new row's id."""
    cur = conn.execute(
        """
        INSERT INTO backtest_results (
            ticker, model_name, start_date, end_date, train_window_days,
            test_window_days, transaction_cost_pct, num_trades, win_rate,
            strategy_return_pct, benchmark_return_pct, max_drawdown_pct,
            sharpe_ratio, detail_json
        ) VALUES (
            :ticker, :model_name, :start_date, :end_date, :train_window_days,
            :test_window_days, :transaction_cost_pct, :num_trades, :win_rate,
            :strategy_return_pct, :benchmark_return_pct, :max_drawdown_pct,
            :sharpe_ratio, :detail_json
        )
        """,
        result,
    )
    return int(cur.lastrowid)
 
 
def get_backtest_results(conn: sqlite3.Connection, ticker: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM backtest_results WHERE ticker = ? ORDER BY created_at DESC", (ticker,)
    ).fetchall()
 
 
def start_pipeline_run(conn: sqlite3.Connection) -> int:
    """Record the start of a pipeline run and return its id."""
    cur = conn.execute("INSERT INTO pipeline_runs (status) VALUES ('running')")
    return int(cur.lastrowid)
 
 
def finish_pipeline_run(conn: sqlite3.Connection, run_id: int, status: str, detail: str | None = None) -> None:
    conn.execute(
        "UPDATE pipeline_runs SET finished_at = datetime('now'), status = ?, detail = ? WHERE id = ?",
        (status, detail, run_id),
    )
 
 
# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
 
def get_latest_price_date(conn: sqlite3.Connection, ticker: str) -> str | None:
    """Return the most recent date we have a price bar for, or None if we have nothing yet."""
    row = conn.execute(
        "SELECT MAX(date) AS max_date FROM prices WHERE ticker = ?", (ticker,)
    ).fetchone()
    return row["max_date"] if row else None
 
 
def get_price_history(conn: sqlite3.Connection, ticker: str, limit: int | None = None) -> list[sqlite3.Row]:
    """Return price bars for a ticker in ascending date order. If `limit` is given,
    return only the most recent `limit` bars (still ascending)."""
    if limit is not None:
        rows = conn.execute(
            "SELECT * FROM prices WHERE ticker = ? ORDER BY date DESC LIMIT ?",
            (ticker, limit),
        ).fetchall()
        return list(reversed(rows))
    return conn.execute(
        "SELECT * FROM prices WHERE ticker = ? ORDER BY date ASC", (ticker,)
    ).fetchall()
 
 
def get_recent_news(conn: sqlite3.Connection, ticker: str, limit: int = 20) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM news WHERE ticker = ? ORDER BY COALESCE(published_at, fetched_at) DESC LIMIT ?",
        (ticker, limit),
    ).fetchall()
 
 
def get_recent_reddit_posts(conn: sqlite3.Connection, ticker: str, limit: int = 20) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM reddit_posts WHERE ticker = ? ORDER BY COALESCE(created_utc, fetched_at) DESC LIMIT ?",
        (ticker, limit),
    ).fetchall()
 
 
if __name__ == "__main__":
    # Running `python database.py` directly just (re)initializes the schema.
    init_db()
    print(f"Database ready at {config.DB_PATH}")
 
