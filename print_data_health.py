"""Print a concise data-fetch health summary for the StockOracle database."""

from __future__ import annotations

import database
import config


def main() -> None:
    database.init_db()
    with database.get_connection() as conn:
        tracked = {
            *config.WATCHLIST,
            *(row[0] for row in conn.execute("SELECT ticker FROM data_health").fetchall()),
        }
        total = len(tracked)
        fresh = conn.execute(
            "SELECT COUNT(*) FROM data_health WHERE last_price_fetch >= datetime('now', '-1 day')"
        ).fetchone()[0]
        fresh_three_days = conn.execute(
            """
            SELECT COUNT(*) FROM data_health
            WHERE last_price_fetch >= datetime('now', '-3 days')
            """
        ).fetchone()[0]
        stale = total - fresh_three_days
        failures = conn.execute(
            """
            SELECT ticker, consecutive_failures, last_error
            FROM data_health
            WHERE consecutive_failures > 0
            ORDER BY consecutive_failures DESC, updated_at DESC
            LIMIT 10
            """
        ).fetchall()

    print(f"Total tickers: {total}")
    print(f"Fresh (<24h): {fresh}")
    print(f"Stale (>3d): {stale}")
    print("Recent failures:")
    if not failures:
        print("  None")
    for row in failures:
        print(f"  {row['ticker']}: {row['consecutive_failures']} - {row['last_error'] or 'unknown error'}")


if __name__ == "__main__":
    main()