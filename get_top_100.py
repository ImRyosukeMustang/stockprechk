"""Download the current S&P 100 ticker list from Wikipedia."""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/S%26P_100"
OUTPUT_PATH = Path(__file__).resolve().parent / "top_100_tickers.txt"
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9-]{0,5}$")


def _normalise_ticker(value: object) -> str | None:
    if pd.isna(value):
        return None
    ticker = str(value).strip().upper().replace(".", "-").replace("/", "-")
    ticker = re.sub(r"[^A-Z0-9-]", "", ticker)
    return ticker if TICKER_PATTERN.fullmatch(ticker) else None


def get_sp100_tickers() -> list[str]:
    """Find and normalize the S&P 100 symbol column across Wikipedia tables."""
    response = requests.get(
        WIKIPEDIA_URL,
        headers={"User-Agent": "StockOracle/1.0 (ticker-list script)"},
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    candidates: list[str] = []
    for table in tables:
        for column in table.columns:
            label = " ".join(str(part).lower() for part in column) if isinstance(column, tuple) else str(column).lower()
            if "symbol" not in label and "ticker" not in label:
                continue
            values = [_normalise_ticker(value) for value in table[column].tolist()]
            values = [value for value in values if value is not None]
            if len(values) >= 80:
                candidates = values
                break
        if candidates:
            break

    if not candidates:
        raise RuntimeError("Could not find an S&P 100 ticker column on the Wikipedia page")

    tickers = list(dict.fromkeys(candidates))
    if not 90 <= len(tickers) <= 110:
        raise RuntimeError(f"Expected approximately 100 tickers, found {len(tickers)}")
    return tickers


def main() -> None:
    tickers = get_sp100_tickers()
    OUTPUT_PATH.write_text("\n".join(tickers) + "\n", encoding="utf-8")
    print(f"Saved {len(tickers)} tickers to {OUTPUT_PATH}")
    print("First 10:", ", ".join(tickers[:10]))


if __name__ == "__main__":
    main()
