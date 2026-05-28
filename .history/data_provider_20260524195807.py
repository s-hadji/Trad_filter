"""
Data layer — MT5 connection, OHLCV cache, indicator computation with cache.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

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


# ── Timeframe constants ────────────────────────────────────────────────────────

TF_MAP: Dict[str, int] = {}
if MT5_AVAILABLE:
    TF_MAP = {
        "M1":  mt5.TIMEFRAME_M1,  "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,  "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,  "W1":  mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
else:
    TF_MAP = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440, "W1": 10080, "MN1": 43200,
    }

TF_ORDER: List[str] = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]


def tf_minutes(tf: str) -> int:
    return {"M1":1,"M5":5,"M15":15,"M30":30,"H1":60,"H4":240,"D1":1440,"W1":10080,"MN1":43200}.get(tf, 1)


# ── Cache entries ──────────────────────────────────────────────────────────────

@dataclass
class OHLCVEntry:
    df: pd.DataFrame
    fetched_at: float = field(default_factory=time.time)


# ── DataProvider ───────────────────────────────────────────────────────────────

class DataProvider:
    def __init__(self) -> None:
        self._connected = False
        # OHLCV cache: (symbol, tf, nb_bars) → OHLCVEntry
        self._ohlcv_cache: Dict[Tuple[str, str, int], OHLCVEntry] = {}
        self._cache_ttl = 5.0
        # Indicator cache: (df_id, df_len, indicator, params_frozen) → Series
        self._ind_cache: Dict[Tuple[int, int, str, FrozenSet], pd.Series] = {}

    # ── Connection ────────────────────────────────────────────────────────────

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

    # ── OHLCV ─────────────────────────────────────────────────────────────────

    def get_ohlcv(
        self, symbol: str, timeframe: str,
        nb_bars: int = 500, force_refresh: bool = False,
    ) -> pd.DataFrame:
        key = (symbol, timeframe, nb_bars)
        now = time.time()
        if not force_refresh and key in self._ohlcv_cache:
            entry = self._ohlcv_cache[key]
            if now - entry.fetched_at < self._cache_ttl:
                return entry.df

        df = self._fetch(symbol, timeframe, nb_bars)
        self._ohlcv_cache[key] = OHLCVEntry(df=df, fetched_at=now)
        return df

    def _fetch(self, symbol: str, timeframe: str, nb_bars: int) -> pd.DataFrame:
        if not self._connected or not MT5_AVAILABLE:
            return self._mock(symbol, timeframe, nb_bars)
        tf_val = TF_MAP.get(timeframe)
        if tf_val is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        rates = mt5.copy_rates_from_pos(symbol, tf_val, 0, nb_bars)
        if rates is None or len(rates) == 0:
            return pd.DataFrame(columns=["time","open","high","low","close","volume"])
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[["time","open","high","low","close","tick_volume"]].copy()
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _mock(self, symbol: str, timeframe: str, nb_bars: int) -> pd.DataFrame:
        rng  = np.random.default_rng(seed=hash(symbol + timeframe) % (2**32))
        mins = tf_minutes(timeframe)
        end  = pd.Timestamp.now().floor("min")
        times = pd.date_range(end=end, periods=nb_bars, freq=f"{mins}min")
        close  = 1.1000 + np.cumsum(rng.normal(0, 0.0003, nb_bars))
        high   = close + rng.uniform(0.0001, 0.0015, nb_bars)
        low    = close - rng.uniform(0.0001, 0.0015, nb_bars)
        open_  = low + rng.uniform(0, 1, nb_bars) * (high - low)
        volume = rng.integers(100, 5000, nb_bars).astype(float)
        return pd.DataFrame({"time":times,"open":open_,"high":high,"low":low,"close":close,"volume":volume})

    # ── Date-range OHLCV ──────────────────────────────────────────────────────

    def get_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        from_date: "pd.Timestamp",
        to_date: "pd.Timestamp",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch bars between *from_date* and *to_date* (inclusive).

        Uses ``mt5.copy_rates_range`` when connected, falls back to synthetic
        data for offline / demo mode.  Results are cached the same way as
        ``get_ohlcv``.
        """
        key = (symbol, timeframe, "range", str(from_date.date()), str(to_date.date()))
        now = time.time()
        if not force_refresh and key in self._ohlcv_cache:
            entry = self._ohlcv_cache[key]
            if now - entry.fetched_at < self._cache_ttl:
                return entry.df

        df = self._fetch_range(symbol, timeframe, from_date, to_date)
        self._ohlcv_cache[key] = OHLCVEntry(df=df, fetched_at=now)
        return df

    def _fetch_range(
        self,
        symbol: str,
        timeframe: str,
        from_date: "pd.Timestamp",
        to_date: "pd.Timestamp",
    ) -> pd.DataFrame:
        if not self._connected or not MT5_AVAILABLE:
            return self._mock_range(symbol, timeframe, from_date, to_date)
        tf_val = TF_MAP.get(timeframe)
        if tf_val is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        rates = mt5.copy_rates_range(
            symbol, tf_val,
            from_date.to_pydatetime(),
            to_date.to_pydatetime(),
        )
        if rates is None or len(rates) == 0:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[["time", "open", "high", "low", "close", "tick_volume"]].copy()
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _mock_range(
        self,
        symbol: str,
        timeframe: str,
        from_date: "pd.Timestamp",
        to_date: "pd.Timestamp",
    ) -> pd.DataFrame:
        rng  = np.random.default_rng(seed=hash(symbol + timeframe) % (2**32))
        mins = tf_minutes(timeframe)
        times = pd.date_range(start=from_date, end=to_date, freq=f"{mins}min")
        n = len(times)
        if n == 0:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        close  = 1.1000 + np.cumsum(rng.normal(0, 0.0003, n))
        high   = close + rng.uniform(0.0001, 0.0015, n)
        low    = close - rng.uniform(0.0001, 0.0015, n)
        open_  = low + rng.uniform(0, 1, n) * (high - low)
        volume = rng.integers(100, 5000, n).astype(float)
        return pd.DataFrame({
            "time": times, "open": open_, "high": high,
            "low": low, "close": close, "volume": volume,
        })

    # ── Indicators (with cache) ────────────────────────────────────────────────

    def compute_indicator(self, df: pd.DataFrame, indicator: str, **params) -> pd.Series:
        ind = indicator.upper()

        # Price series — no computation needed
        price_map = {"CLOSE":"close","OPEN":"open","HIGH":"high","LOW":"low","VOLUME":"volume"}
        if ind in price_map:
            return df[price_map[ind]].copy()

        # Cache key: stable as long as DataProvider's OHLCV cache returns the same df object
        cache_key = (id(df), len(df), ind, frozenset(params.items()))
        if cache_key in self._ind_cache:
            return self._ind_cache[cache_key]

        result = (
            self._compute_ta(df, ind, **params)
            if TA_AVAILABLE
            else self._compute_fallback(df, ind, **params)
        )
        self._ind_cache[cache_key] = result
        return result

    def _compute_ta(self, df: pd.DataFrame, ind: str, **p) -> pd.Series:
        period = int(p.get("period", 14))
        nan    = pd.Series(np.nan, index=df.index)

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
            fast = int(p.get("fast", 12)); slow = int(p.get("slow", 26)); sig = int(p.get("signal", 9))
            m = _ta.trend.MACD(df["close"], window_fast=fast, window_slow=slow, window_sign=sig)
            return m.macd_diff()
        if ind in ("BBANDS_UPPER", "BBANDS_MID", "BBANDS_LOWER"):
            std = float(p.get("std", 2.0))
            bb  = _ta.volatility.BollingerBands(df["close"], window=period, window_dev=std)
            return {"BBANDS_UPPER": bb.bollinger_hband,
                    "BBANDS_MID":   bb.bollinger_mavg,
                    "BBANDS_LOWER": bb.bollinger_lband}[ind]()
        if ind in ("STOCH_K", "STOCH_D"):
            k = int(p.get("k", 14)); d = int(p.get("d", 3))
            st = _ta.momentum.StochasticOscillator(
                df["high"], df["low"], df["close"], window=k, smooth_window=d)
            return st.stoch() if ind == "STOCH_K" else st.stoch_signal()
        raise ValueError(f"Unknown indicator: {ind}")

    def _compute_fallback(self, df: pd.DataFrame, ind: str, **p) -> pd.Series:
        period = int(p.get("period", 14))
        if ind == "EMA":
            return df["close"].ewm(span=period, adjust=False).mean()
        if ind == "SMA":
            return df["close"].rolling(period).mean()
        if ind == "RSI":
            delta = df["close"].diff()
            gain  = delta.clip(lower=0).rolling(period).mean()
            loss  = (-delta.clip(upper=0)).rolling(period).mean()
            return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        if ind == "ATR":
            hl = df["high"] - df["low"]
            hc = (df["high"] - df["close"].shift()).abs()
            lc = (df["low"]  - df["close"].shift()).abs()
            return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()
        return pd.Series(np.nan, index=df.index)

    # ── Cache management ───────────────────────────────────────────────────────

    def set_cache_ttl(self, seconds: float) -> None:
        self._cache_ttl = seconds

    def invalidate_cache(self, symbol: Optional[str] = None) -> None:
        if symbol is None:
            self._ohlcv_cache.clear()
        else:
            for k in [k for k in self._ohlcv_cache if k[0] == symbol]:
                del self._ohlcv_cache[k]
        # Indicator cache may reference stale DataFrames — clear it too
        self._ind_cache.clear()
