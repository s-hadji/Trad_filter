"""
Shared fixtures: synthetic OHLCV data and a mock DataProvider.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_provider import DataProvider


def make_ohlcv(
    n: int = 100,
    start: str = "2024-01-01 00:00",
    freq_minutes: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(start=start, periods=n, freq=f"{freq_minutes}min")
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0003, n))
    high = close + rng.uniform(0.0001, 0.001, n)
    low  = close - rng.uniform(0.0001, 0.001, n)
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(100, 3000, n).astype(float)
    return pd.DataFrame({"time": times, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


class MockProvider(DataProvider):
    """DataProvider that never calls MT5 — always serves synthetic data."""

    def __init__(self, data_map=None):
        super().__init__()
        self._map = data_map or {}

    def get_ohlcv(self, symbol, timeframe, nb_bars=500, force_refresh=False):
        if (symbol, timeframe) in self._map:
            return self._map[(symbol, timeframe)]
        return make_ohlcv(nb_bars, freq_minutes={"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60}.get(timeframe, 15))

    def compute_indicator(self, df, indicator, **params):
        return super()._fallback_indicator(df, indicator.upper(), **params)


@pytest.fixture
def m15_df():
    return make_ohlcv(200, freq_minutes=15)


@pytest.fixture
def m5_df():
    return make_ohlcv(600, freq_minutes=5)


@pytest.fixture
def h1_df():
    return make_ohlcv(100, freq_minutes=60)


@pytest.fixture
def provider():
    return MockProvider()
