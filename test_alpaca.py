from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

from alpaca_keys import ALPACA_API_KEY, ALPACA_SECRET_KEY

client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

print("--- Latest Quote for AAPL ---")
quote = client.get_stock_latest_quote(
    StockLatestQuoteRequest(symbol_or_symbols=["AAPL"], feed=DataFeed.IEX)
)
q = quote["AAPL"]
print(f"Bid: ${q.bid_price}")
print(f"Ask: ${q.ask_price}")

print("\n--- Last 3 Hourly Bars for AAPL ---")
req = StockBarsRequest(
    symbol_or_symbols=["AAPL"],
    timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Hour),
    start=datetime.now(ZoneInfo("America/New_York")) - timedelta(days=3),
    feed=DataFeed.IEX,
)
bars = client.get_stock_bars(req).df
print(bars.tail(3))
