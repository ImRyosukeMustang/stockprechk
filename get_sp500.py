"""Download the current S&P 500 ticker list from Wikipedia."""

from pathlib import Path

import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUTPUT = Path(__file__).resolve().parent / "sp500_tickers.txt"


def get_sp500_tickers() -> list[str]:
    """Scrape, normalize, and sort the current S&P 500 ticker symbols."""
    table = pd.read_html(WIKI_URL)[0]
    return sorted({str(symbol).strip().replace(".", "-") for symbol in table["Symbol"]})


def main() -> None:
    tickers = get_sp500_tickers()
    OUTPUT.write_text("\n".join(tickers) + "\n", encoding="utf-8")
    print(f"Saved {len(tickers)} tickers to {OUTPUT}")


if __name__ == "__main__":
    main()