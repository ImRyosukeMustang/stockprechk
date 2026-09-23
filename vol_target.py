"""Position sizing via volatility targeting."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

log = config.get_logger(__name__)


def compute_position_size(
    closes: pd.Series, target_vol: float = 0.15, lookback: int = 20
) -> float:
    """Scale exposure inversely to recent annualized realized volatility."""
    returns = closes.pct_change().dropna()
    if len(returns) < lookback:
        return 1.0

    realized_vol = float(returns.iloc[-lookback:].std() * np.sqrt(252))
    if pd.isna(realized_vol) or realized_vol == 0.0:
        return 1.0
    return float(min(1.0, target_vol / realized_vol))


if __name__ == "__main__":
    dummy_closes = pd.Series([
        100, 102, 101, 103, 100, 99, 101, 105, 104, 106,
        108, 107, 109, 110, 108, 107, 109, 111, 110, 112, 115,
    ])
    size = compute_position_size(dummy_closes, target_vol=0.15, lookback=20)
    log.info("Standalone test position size: %.2f%%", size * 100)