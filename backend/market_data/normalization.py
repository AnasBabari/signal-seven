"""Strict provider-independent OHLCV normalization with auditable repairs."""

from __future__ import annotations

import hashlib
import logging
import math

import numpy as np
import pandas as pd

from .base import BarRepairRecord, MarketDataProviderError

logger = logging.getLogger(__name__)

REQUIRED_OHLCV = ("Open", "High", "Low", "Close", "Volume")
MIN_ROWS_FOR_REPAIR = 60
TRAILING_PROTECTED_SESSIONS = 20
MAX_REPAIR_FRACTION = 0.0025
MAX_REPAIR_DATES_CAP = 5
MAX_REPAIR_RELATIVE_DRIFT = 0.05


def prepare_daily_index(
    frame: pd.DataFrame,
    *,
    provider: str,
    symbol: str,
    timestamp_kind: str = "session_label",
) -> pd.DataFrame:
    """Parse timestamps, deduplicate/sort index, and coerce numeric OHLCV columns."""
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
    return data


def raw_frame_fingerprint(frame: pd.DataFrame, *, provider: str, symbol: str) -> str:
    """Deterministic SHA-256 of the trimmed pre-repair OHLCV frame."""
    digest = hashlib.sha256()
    digest.update(provider.encode("utf-8"))
    digest.update(b"|")
    digest.update(symbol.upper().encode("utf-8"))
    digest.update(b"|")
    digest.update("|".join(REQUIRED_OHLCV).encode("utf-8"))
    digest.update(
        pd.util.hash_pandas_object(frame.loc[:, list(REQUIRED_OHLCV)], index=True).values.tobytes()
    )
    return digest.hexdigest()


def normalize_daily_bars_with_audit(
    frame: pd.DataFrame,
    *,
    provider: str,
    symbol: str,
    timestamp_kind: str = "session_label",
    max_session: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, str, tuple[BarRepairRecord, ...]]:
    """Trim to completed sessions, validate, apply conservative auditable repairs, and return audit trail."""
    data = prepare_daily_index(
        frame, provider=provider, symbol=symbol, timestamp_kind=timestamp_kind
    )
    if max_session is not None:
        cutoff = pd.Timestamp(max_session).tz_localize(None).normalize()
        data = data.loc[data.index <= cutoff].copy()
    if data.empty:
        raise MarketDataProviderError(f"{provider} returned no completed bars for {symbol}")

    raw_fp = raw_frame_fingerprint(data, provider=provider, symbol=symbol)
    n_rows = len(data)
    max_affected_dates = (
        0
        if n_rows < MIN_ROWS_FOR_REPAIR
        else min(MAX_REPAIR_DATES_CAP, max(1, math.floor(n_rows * MAX_REPAIR_FRACTION)))
    )
    protected_start = max(0, n_rows - TRAILING_PROTECTED_SESSIONS)
    protected_dates = set(data.index[protected_start:])

    values = data.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise MarketDataProviderError(f"{provider} returned infinite OHLCV data for {symbol}")

    affected_dates: set[str] = set()
    records: list[BarRepairRecord] = []
    drop_indices: list[pd.Timestamp] = []

    any_nan_mask = data.isna().any(axis=1)
    for ts in data.index[any_nan_mask]:
        ohlc_all_nan = bool(data.loc[ts, ["Open", "High", "Low", "Close"]].isna().all())
        vol_val = data.at[ts, "Volume"]
        vol_empty = bool(pd.isna(vol_val) or float(vol_val) == 0.0)
        if not (ohlc_all_nan and vol_empty):
            raise MarketDataProviderError(f"{provider} returned non-finite OHLCV data for {symbol}")
        if ts in protected_dates or max_affected_dates == 0:
            raise MarketDataProviderError(f"{provider} returned non-finite OHLCV data for {symbol}")
        date_iso = ts.date().isoformat()
        affected_dates.add(date_iso)
        drop_indices.append(ts)
        records.append(
            BarRepairRecord(
                date=date_iso,
                field="ROW",
                original=None,
                repaired=None,
                reason="dropped_empty_nan_session",
            )
        )

    if len(affected_dates) > max_affected_dates:
        raise MarketDataProviderError(f"{provider} returned non-finite OHLCV data for {symbol}")
    if drop_indices:
        data = data.drop(index=drop_indices)
    if data.empty or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise MarketDataProviderError(f"{provider} returned non-finite OHLCV data for {symbol}")

    # Validate positivity BEFORE any envelope clamping.
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise MarketDataProviderError(f"{provider} returned non-positive prices for {symbol}")
    if (data["Volume"] < 0).any():
        raise MarketDataProviderError(f"{provider} returned negative volume for {symbol}")

    oc_max = data[["Open", "Close"]].max(axis=1)
    oc_min = data[["Open", "Close"]].min(axis=1)
    bad_mask = (data["High"] < oc_max) | (data["Low"] > oc_min)

    if bad_mask.any():
        for col in ("Open", "High", "Low", "Close"):
            data[col] = data[col].astype(float)
        for ts in data.index[bad_mask]:
            if ts in protected_dates or max_affected_dates == 0:
                raise MarketDataProviderError(
                    f"{provider} returned impossible OHLC bars for {symbol}"
                )
            date_iso = ts.date().isoformat()
            denom = float(oc_max.loc[ts])
            high_val = float(data.at[ts, "High"])
            if high_val < denom:
                drift = abs(denom - high_val) / denom
                if drift > MAX_REPAIR_RELATIVE_DRIFT:
                    raise MarketDataProviderError(
                        f"{provider} returned impossible OHLC bars for {symbol}"
                    )
                data.at[ts, "High"] = denom
                affected_dates.add(date_iso)
                records.append(
                    BarRepairRecord(
                        date=date_iso,
                        field="High",
                        original=high_val,
                        repaired=denom,
                        reason="clamped_high_below_open_close",
                    )
                )
            min_oc = float(oc_min.loc[ts])
            low_val = float(data.at[ts, "Low"])
            if low_val > min_oc:
                drift = abs(low_val - min_oc) / denom
                if drift > MAX_REPAIR_RELATIVE_DRIFT:
                    raise MarketDataProviderError(
                        f"{provider} returned impossible OHLC bars for {symbol}"
                    )
                data.at[ts, "Low"] = min_oc
                affected_dates.add(date_iso)
                records.append(
                    BarRepairRecord(
                        date=date_iso,
                        field="Low",
                        original=low_val,
                        repaired=min_oc,
                        reason="clamped_low_above_open_close",
                    )
                )
        if len(affected_dates) > max_affected_dates:
            raise MarketDataProviderError(f"{provider} returned impossible OHLC bars for {symbol}")

    # Final invariant verification.
    if (
        data.empty
        or not np.isfinite(data.to_numpy(dtype=float)).all()
        or (data[["Open", "High", "Low", "Close"]] <= 0).any().any()
        or (data["Volume"] < 0).any()
        or (
            (data["Low"] > data[["Open", "Close"]].min(axis=1))
            | (data["High"] < data[["Open", "Close"]].max(axis=1))
        ).any()
    ):
        raise MarketDataProviderError(f"{provider} returned impossible OHLC bars for {symbol}")

    audit = tuple(records)
    for rec in audit:
        logger.warning(
            "market_data_repair provider=%s symbol=%s date=%s field=%s original=%s repaired=%s reason=%s raw_fp=%s",
            provider,
            symbol.upper(),
            rec.date,
            rec.field,
            rec.original,
            rec.repaired,
            rec.reason,
            raw_fp[:12],
        )

    data.attrs["raw_source_fingerprint"] = raw_fp
    data.attrs["data_quality_audit"] = audit
    return data, raw_fp, audit


def normalize_daily_bars(
    frame: pd.DataFrame, *, provider: str, symbol: str, timestamp_kind: str = "session_label"
) -> pd.DataFrame:
    """Return a finite, ordered, timezone-naive daily OHLCV frame."""
    normalized, _, _ = normalize_daily_bars_with_audit(
        frame, provider=provider, symbol=symbol, timestamp_kind=timestamp_kind
    )
    return normalized


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
