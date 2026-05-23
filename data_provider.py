"""
Data layer: MT5 connection, OHLCV retrieval, indicator computation, caching.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    import ta as _ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Timeframe constants (MT5 values)
# ---------------------------------------------------------------------------

TF_MAP: Dict[str, int] = {}

if MT5_AVAILABLE:
    TF_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
else:
    # Fallback values for development/testing without MT5
    TF_MAP = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440, "W1": 10080, "MN1": 43200,
    }

# Ordering for cross-TF logic (ascending = smallest first)
TF_ORDER: List[str] = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]


def tf_minutes(tf: str) -> int:
    """Return approximate duration of one bar in minutes."""
    mapping = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440, "W1": 10080, "MN1": 43200,
    }
    return mapping.get(tf, 1)


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    df: pd.DataFrame
    fetched_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# DataProvider
# ---------------------------------------------------------------------------

class DataProvider:
    """
    Manages MT5 connection and provides OHLCV + indicator data with caching.
    """

    def __init__(self) -> None:
        self._connected = False
        # cache key: (symbol, timeframe_str, nb_bars)
        self._cache: Dict[Tuple[str, str, int], CacheEntry] = {}
        self._cache_ttl = 5.0  # seconds

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            return False
        if not mt5.initialize():
            return False
        self._connected = True
        return True

    def disconnect(self) -> None:
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def available_symbols(self) -> List[str]:
        if not self._connected:
            return []
        symbols = mt5.symbols_get()
        return [s.name for s in symbols] if symbols else []

    # ------------------------------------------------------------------
    # OHLCV retrieval
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        nb_bars: int = 500,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return OHLCV DataFrame (columns: time, open, high, low, close, volume).
        Cached per (symbol, timeframe, nb_bars) with TTL.
        """
        key = (symbol, timeframe, nb_bars)
        now = time.time()

        if not force_refresh and key in self._cache:
            entry = self._cache[key]
            if now - entry.fetched_at < self._cache_ttl:
                return entry.df

        df = self._fetch_from_mt5(symbol, timeframe, nb_bars)
        self._cache[key] = CacheEntry(df=df, fetched_at=now)
        return df

    def _fetch_from_mt5(
        self, symbol: str, timeframe: str, nb_bars: int
    ) -> pd.DataFrame:
        if not self._connected or not MT5_AVAILABLE:
            return self._generate_mock_data(symbol, timeframe, nb_bars)

        tf_val = TF_MAP.get(timeframe)
        if tf_val is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        rates = mt5.copy_rates_from_pos(symbol, tf_val, 0, nb_bars)
        if rates is None or len(rates) == 0:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[["time", "open", "high", "low", "close", "tick_volume"]].copy()
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _generate_mock_data(
        self, symbol: str, timeframe: str, nb_bars: int
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for development without MT5."""
        rng = np.random.default_rng(seed=hash(symbol + timeframe) % (2**32))
        minutes = tf_minutes(timeframe)
        end = pd.Timestamp.now().floor("min")
        times = pd.date_range(end=end, periods=nb_bars, freq=f"{minutes}min")

        close = 1.1000 + np.cumsum(rng.normal(0, 0.0003, nb_bars))
        high = close + rng.uniform(0.0001, 0.0015, nb_bars)
        low = close - rng.uniform(0.0001, 0.0015, nb_bars)
        open_ = low + rng.uniform(0, 1, nb_bars) * (high - low)
        volume = rng.integers(100, 5000, nb_bars).astype(float)

        return pd.DataFrame({
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    # ------------------------------------------------------------------
    # Indicator computation
    # ------------------------------------------------------------------

    def compute_indicator(
        self,
        df: pd.DataFrame,
        indicator: str,
        **params,
    ) -> pd.Series:
        """
        Compute a named indicator on a OHLCV DataFrame.
        Supported: RSI, EMA, SMA, MACD_hist, ATR, BBANDS_upper/lower/mid,
                   STOCH_k/d, Close, Open, High, Low, Volume.
        """
        ind = indicator.upper()

        if ind == "CLOSE":
            return df["close"].copy()
        if ind == "OPEN":
            return df["open"].copy()
        if ind == "HIGH":
            return df["high"].copy()
        if ind == "LOW":
            return df["low"].copy()
        if ind == "VOLUME":
            return df["volume"].copy()

        if not TA_AVAILABLE:
            return self._fallback_indicator(df, ind, **params)

        period = int(params.get("period", 14))

        if ind == "RSI":
            return _ta.momentum.RSIIndicator(df["close"], window=period).rsi()

        if ind == "EMA":
            return _ta.trend.EMAIndicator(df["close"], window=period).ema_indicator()

        if ind == "SMA":
            return _ta.trend.SMAIndicator(df["close"], window=period).sma_indicator()

        if ind == "ATR":
            return _ta.volatility.AverageTrueRange(
                df["high"], df["low"], df["close"], window=period
            ).average_true_range()

        if ind == "MACD_HIST":
            fast   = int(params.get("fast", 12))
            slow   = int(params.get("slow", 26))
            signal = int(params.get("signal", 9))
            macd = _ta.trend.MACD(df["close"], window_fast=fast, window_slow=slow, window_sign=signal)
            return macd.macd_diff()

        if ind in ("BBANDS_UPPER", "BBANDS_MID", "BBANDS_LOWER"):
            std = float(params.get("std", 2.0))
            bb = _ta.volatility.BollingerBands(df["close"], window=period, window_dev=std)
            if ind == "BBANDS_UPPER":
                return bb.bollinger_hband()
            if ind == "BBANDS_LOWER":
                return bb.bollinger_lband()
            return bb.bollinger_mavg()

        if ind in ("STOCH_K", "STOCH_D"):
            k = int(params.get("k", 14))
            d = int(params.get("d", 3))
            stoch = _ta.momentum.StochasticOscillator(
                df["high"], df["low"], df["close"], window=k, smooth_window=d
            )
            return stoch.stoch() if ind == "STOCH_K" else stoch.stoch_signal()

        raise ValueError(f"Unknown indicator: {indicator}")

    def _fallback_indicator(
        self, df: pd.DataFrame, ind: str, **params
    ) -> pd.Series:
        """Minimal pure-pandas fallback when pandas-ta is unavailable."""
        period = int(params.get("period", 14))

        if ind == "EMA":
            return df["close"].ewm(span=period, adjust=False).mean()
        if ind == "SMA":
            return df["close"].rolling(period).mean()
        if ind == "RSI":
            delta = df["close"].diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs))
        if ind == "ATR":
            hl = df["high"] - df["low"]
            hc = (df["high"] - df["close"].shift()).abs()
            lc = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            return tr.rolling(period).mean()

        return pd.Series(np.nan, index=df.index)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def set_cache_ttl(self, seconds: float) -> None:
        self._cache_ttl = seconds

    def invalidate_cache(self, symbol: Optional[str] = None) -> None:
        if symbol is None:
            self._cache.clear()
        else:
            keys = [k for k in self._cache if k[0] == symbol]
            for k in keys:
                del self._cache[k]
