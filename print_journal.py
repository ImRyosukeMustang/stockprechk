"""Print the SMA signal journal summary."""

from __future__ import annotations

import database
from journal import backfill_forward_returns, print_journal_summary


def main() -> None:
    database.init_db()
    with database.get_connection() as conn:
        backfill_forward_returns(conn)
        print_journal_summary(conn)


if __name__ == "__main__":
    main()