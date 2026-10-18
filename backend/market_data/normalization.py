"""Strict provider-independent OHLCV normalization."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .base import MarketDataProviderError

REQUIRED_OHLCV = ("Open", "High", "Low", "Close", "Volume")


def normalize_daily_bars(
    frame: pd.DataFrame, *, provider: str, symbol: str, timestamp_kind: str = "session_label"
) -> pd.DataFrame:
    """Return a finite, ordered, timezone-naive daily OHLCV frame."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise MarketDataProviderError(f"{provider} returned no bars for {symbol}")
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    missing = set(REQUIRED_OHLCV).difference(data.columns)
    if missing:
        raise MarketDataProviderError(
            f"{provider} returned an incomplete OHLCV schema for {symbol}"
        )
    if timestamp_kind == "instant":
        index = pd.to_datetime(data.index, errors="coerce", utc=True)
        timezone = "Europe/London" if symbol.upper().endswith(".L") else "America/New_York"
        index = index.tz_convert(timezone).tz_localize(None)
    elif timestamp_kind == "session_label":
        index = pd.to_datetime(data.index, errors="coerce").tz_localize(None)
    else:
        raise ValueError("Unknown daily timestamp semantics")
    if index.isna().any():
        raise MarketDataProviderError(f"{provider} returned invalid timestamps for {symbol}")
    data.index = index.normalize()
    data = data.loc[~data.index.duplicated(keep="last")].sort_index()
    data = data.loc[:, list(REQUIRED_OHLCV)].apply(pd.to_numeric, errors="coerce")
    if data.empty or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise MarketDataProviderError(f"{provider} returned non-finite OHLCV data for {symbol}")
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise MarketDataProviderError(f"{provider} returned non-positive prices for {symbol}")
    if (data["Volume"] < 0).any():
        raise MarketDataProviderError(f"{provider} returned negative volume for {symbol}")
    if (
        (data["Low"] > data[["Open", "Close"]].min(axis=1))
        | (data["High"] < data[["Open", "Close"]].max(axis=1))
    ).any():
        raise MarketDataProviderError(f"{provider} returned impossible OHLC bars for {symbol}")
    return data


def daily_frame_fingerprint(frame: pd.DataFrame) -> str:
    """Identity shared by history, prediction and scoring for the full input."""
    digest = hashlib.sha256()
    digest.update("|".join(REQUIRED_OHLCV).encode())
    digest.update(
        pd.util.hash_pandas_object(frame.loc[:, list(REQUIRED_OHLCV)], index=True).values.tobytes()
    )
    for key in ("data_provider", "data_feed", "price_adjustment"):
        digest.update(str(frame.attrs.get(key, "unknown")).encode())
    return digest.hexdigest()
