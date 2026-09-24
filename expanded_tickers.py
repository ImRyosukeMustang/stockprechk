"""Build StockOracle's expanded ticker universe from major indexes and priorities.

The script scrapes S&P 500, Nasdaq 100, and Russell 1000 constituents when
available, combines them with curated priority symbols, filters malformed
values, and writes the sorted result to ``universe_tickers.txt``.
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

import config

log = config.get_logger(__name__)

OUTPUT_PATH = Path(__file__).resolve().parent / "universe_tickers.txt"
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_HEADERS = {"User-Agent": "StockOracle/1.0 (ticker-universe script)"}

INDEX_URLS = {
    "S&P 500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",),
    "Nasdaq 100": (
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
    ),
    "Russell 1000": (
        "https://en.wikipedia.org/wiki/Russell_1000_Index",
        "https://en.wikipedia.org/wiki/List_of_Russell_1000_companies",
    ),
}

PRIORITY_ADRS = [
    "SONY", "TTWO", "NTDOY", "TM", "HMC", "MUFG", "CAJ", "KB", "SHG", "BABA",
    "JD", "PDD", "NTES", "BIDU", "TCEHY", "TSM", "UMC", "ASX", "CHT", "SAP",
    "ASML", "SHEL", "BP", "TTE", "SNY", "AZN", "GSK", "UL", "DEO", "BT", "HSBC",
    "RY", "TD", "BNS", "BMO", "CNI", "CP", "SHOP", "SPOT", "NVO", "AZN", "SNY",
    "RHHBY", "LLY", "NVS", "GSK",
]

PRIORITY_TECH = [
    "PLTR", "SNOW", "CRWD", "ZS", "DDOG", "NET", "MDB", "OKTA", "COIN", "HOOD",
    "SOFI", "AFRM", "UPST", "LC", "OPEN", "RBLX", "U", "SPOT", "SQ", "PYPL", "SHOP",
    "TWLO", "ZM", "DOCU", "ROKU", "PINS", "SNAP",
]

PRIORITY_MEME = ["AMC", "GME", "BB", "NOK", "WISH", "CLOV", "SPCE", "PLTR", "RIVN", "LCID", "NIO", "XPEV", "LI"]

PRIORITY_ETFS = [
    "ARKK", "ARKG", "ARKW", "ARKF", "ARKQ", "ICLN", "TAN", "LIT", "SOXX", "SMH", "XLE",
    "XLF", "XLK", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLY", "VTI", "VOO", "IVV",
    "IWM", "QQQ", "DIA", "VEA", "VWO", "EEM", "EFA", "AGG", "BND", "LQD", "HYG", "TLT",
    "IEF", "SHY", "GLD", "SLV", "USO", "UNG", "VNQ",
]


def _clean_tickers(values: list[object]) -> list[str]:
    """Normalize, validate, deduplicate, and sort ticker values."""
    cleaned: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        ticker = str(value).strip().upper().replace(".", "-")
        if ticker and len(ticker) <= 10 and re.fullmatch(r"[A-Z0-9-]+", ticker):
            cleaned.add(ticker)
    return sorted(cleaned)


def _scrape_index(name: str, urls: tuple[str, ...], columns: tuple[str, ...]) -> list[str]:
    """Scrape a ticker column from one Wikipedia index, returning [] on failure."""
    for url in urls:
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
            for table in tables:
                for column in columns:
                    if column in table.columns:
                        return _clean_tickers(table[column].tolist())
            log.warning("%s table structure was unexpected at %s; trying fallback.", name, url)
        except Exception as exc:
            log.warning("Could not scrape %s from %s: %s", name, url, exc)
    log.warning("No usable table found for %s; skipping source.", name)
    return []


def build_universe() -> list[str]:
    """Build, save, and return the expanded ticker universe."""
    sp500 = _scrape_index("S&P 500", INDEX_URLS["S&P 500"], ("Symbol",))
    nasdaq100 = _scrape_index("Nasdaq 100", INDEX_URLS["Nasdaq 100"], ("Ticker", "Symbol"))
    russell1000 = _scrape_index("Russell 1000", INDEX_URLS["Russell 1000"], ("Ticker", "Symbol"))

    priority_groups = {
        "Priority ADRs": _clean_tickers(PRIORITY_ADRS),
        "Priority tech": _clean_tickers(PRIORITY_TECH),
        "Priority meme/retail": _clean_tickers(PRIORITY_MEME),
        "ETFs": _clean_tickers(PRIORITY_ETFS),
    }
    universe = _clean_tickers(sp500 + nasdaq100 + russell1000 + [
        ticker
        for group in priority_groups.values()
        for ticker in group
    ])
    OUTPUT_PATH.write_text("\n".join(universe) + "\n", encoding="utf-8")

    print(f"S&P 500: {len(sp500)} tickers")
    print(f"Nasdaq 100: {len(nasdaq100)} tickers")
    print(f"Russell 1000: {len(russell1000)} tickers")
    print(f"Priority ADRs: {len(priority_groups['Priority ADRs'])} tickers")
    print(f"Priority tech: {len(priority_groups['Priority tech'])} tickers")
    print(f"ETFs: {len(priority_groups['ETFs'])} tickers")
    print("============================")
    print(f"Total unique: {len(universe)} tickers")
    print("Saved to universe_tickers.txt")
    return universe


if __name__ == "__main__":
    build_universe()