"""Optional read-only access to an Alpaca paper account."""

from __future__ import annotations

import os

import config

log = config.get_logger(__name__)

try:
    from alpaca_keys import ALPACA_API_KEY, ALPACA_SECRET_KEY
except ImportError:
    ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

if ALPACA_API_KEY:
    ALPACA_API_KEY = ALPACA_API_KEY.strip()
if ALPACA_SECRET_KEY:
    ALPACA_SECRET_KEY = ALPACA_SECRET_KEY.strip()


def _client():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        log.warning("Alpaca credentials unavailable; paper portfolio is disabled.")
        return None
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        log.warning("alpaca-py is not installed; paper portfolio is disabled.")
        return None
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)


def _number(value):
    return float(value) if value is not None else None


def get_paper_account() -> dict | None:
    """Return paper account equity, cash, buying power, and status."""
    client = _client()
    if client is None:
        return None
    try:
        account = client.get_account()
        return {
            "id": str(account.id),
            "status": str(account.status),
            "equity": _number(account.equity),
            "cash": _number(account.cash),
            "buying_power": _number(account.buying_power),
            "portfolio_value": _number(account.portfolio_value),
        }
    except Exception as exc:
        log.warning("Unable to read Alpaca paper account: %s", exc)
        return None


def get_current_positions() -> list[dict]:
    """Return current paper positions as serializable dictionaries."""
    client = _client()
    if client is None:
        return []
    try:
        positions = client.get_all_positions()
        return [
            {
                "symbol": position.symbol,
                "quantity": _number(position.qty),
                "market_value": _number(position.market_value),
                "cost_basis": _number(position.cost_basis),
                "unrealized_pl": _number(position.unrealized_pl),
                "unrealized_plpc": _number(position.unrealized_plpc),
                "side": str(position.side),
            }
            for position in positions
        ]
    except Exception as exc:
        log.warning("Unable to read Alpaca paper positions: %s", exc)
        return []


def get_order_history(limit: int = 20) -> list[dict]:
    """Return recent paper orders as serializable dictionaries."""
    client = _client()
    if client is None:
        return []
    try:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
        )
        return [
            {
                "id": str(order.id),
                "symbol": order.symbol,
                "side": str(order.side),
                "status": str(order.status),
                "quantity": _number(order.qty),
                "filled_quantity": _number(order.filled_qty),
                "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                "filled_avg_price": _number(order.filled_avg_price),
            }
            for order in orders
        ]
    except Exception as exc:
        log.warning("Unable to read Alpaca paper orders: %s", exc)
        return []
