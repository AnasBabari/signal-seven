"""Session-aware, non-pickle cache for normalized daily bars."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from market_data.base import BarRepairRecord, MarketDataResult
from market_data.normalization import REQUIRED_OHLCV

MAX_CACHE_FILE_BYTES = 25 * 1024 * 1024
CACHE_SCHEMA_VERSION = 3


class MarketDataCache:
    """Atomic JSON cache keyed by provider and symbol."""

    def __init__(self, directory: Path | str | None, *, enabled: bool = True) -> None:
        self.directory = Path(directory) if directory else None
        self.enabled = bool(enabled and self.directory)
        self._lock = threading.RLock()
        if self.enabled and self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _identity(provider: str, symbol: str) -> str:
        raw = f"{provider.lower()}:{symbol.upper()}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _path(self, provider: str, symbol: str) -> Path:
        if self.directory is None:
            raise RuntimeError("market-data cache has no directory")
        return self.directory / f"{self._identity(provider, symbol)}.json"

    def load(
        self, provider: str, symbol: str, *, required_session: str, namespace: str | None = None
    ) -> MarketDataResult | None:
        if not self.enabled:
            return None
        path = self._path(namespace or provider, symbol)
        with self._lock:
            try:
                if path.stat().st_size > MAX_CACHE_FILE_BYTES:
                    return None
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("provider") != provider or payload.get("symbol") != symbol.upper():
                    return None
                if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
                    return None
                data_as_of = str(payload["data_as_of"])
                # A cache entry from a later calendar date can contain an
                # in-progress daily bar when it was created before the close.
                # Only an exact completed-session match is safe to reuse.
                if data_as_of != required_session:
                    return None
                rows = payload["rows"]
                if not isinstance(rows, list) or not rows:
                    return None
                if any(not isinstance(row, list) or len(row) != 6 for row in rows):
                    return None
                index = pd.DatetimeIndex(
                    pd.to_datetime([row[0] for row in rows], format="%Y-%m-%d", errors="raise")
                )
                frame = pd.DataFrame(
                    [row[1:] for row in rows],
                    columns=list(REQUIRED_OHLCV),
                    index=index,
                    dtype=float,
                )
                if (
                    frame.empty
                    or not frame.index.is_monotonic_increasing
                    or frame.index.has_duplicates
                    or frame.index[-1].date().isoformat() != data_as_of
                    or not np.isfinite(frame.to_numpy(dtype=float)).all()
                ):
                    return None
                raw_audit = payload.get("data_quality_audit", ())
                audit = tuple(BarRepairRecord.from_dict(item) for item in raw_audit)
                raw_fp = str(payload.get("raw_source_fingerprint", ""))
            except (OSError, ValueError, KeyError, TypeError):
                return None
        return MarketDataResult(
            frame=frame,
            provider=provider,
            data_as_of=data_as_of,
            cache_status="hit",
            feed=payload.get("feed", "unknown"),
            adjustment=payload.get("adjustment", "unknown"),
            data_fingerprint=payload.get("data_fingerprint", ""),
            raw_source_fingerprint=raw_fp,
            data_quality_audit=audit,
            timestamp_kind="session_label",
            post_close_grace_used=False,
        )

    def save(self, symbol: str, result: MarketDataResult, *, namespace: str | None = None) -> None:
        if not self.enabled:
            return
        path = self._path(namespace or result.provider, symbol)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "feed": result.feed,
            "adjustment": result.adjustment,
            "data_fingerprint": result.data_fingerprint,
            "raw_source_fingerprint": result.raw_source_fingerprint,
            "data_quality_audit": [record.to_dict() for record in result.data_quality_audit],
            "provider": result.provider,
            "symbol": symbol.upper(),
            "data_as_of": result.data_as_of,
            "stored_at": datetime.now(UTC).isoformat(),
            "rows": [
                [index.date().isoformat(), *[float(value) for value in row]]
                for index, row in zip(
                    result.frame.index,
                    result.frame.loc[:, list(REQUIRED_OHLCV)].to_numpy(),
                    strict=True,
                )
            ],
        }
        with self._lock:
            fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, path)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temp_name)

    def fresh_entry_count(
        self,
        *,
        required_session: str,
        providers: set[str] | None = None,
        exclude_lse_symbols: bool = False,
    ) -> int:
        if not self.enabled or self.directory is None:
            return 0
        count = 0
        with self._lock:
            for path in self.directory.glob("*.json"):
                try:
                    if path.stat().st_size > MAX_CACHE_FILE_BYTES:
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
                        continue
                    if str(payload.get("data_as_of", "")) != required_session:
                        continue
                    if providers is not None and payload.get("provider") not in providers:
                        continue
                    if exclude_lse_symbols and str(payload.get("symbol", "")).upper().endswith(
                        ".L"
                    ):
                        continue
                    count += 1
                except (OSError, ValueError, TypeError):
                    continue
        return count
