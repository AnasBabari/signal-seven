"""Session-aware, non-pickle cache for normalized daily bars."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from market_data.base import BarRepairRecord, MarketDataResult
from market_data.normalization import REQUIRED_OHLCV

MAX_CACHE_FILE_BYTES = 25 * 1024 * 1024
CACHE_SCHEMA_VERSION = 3
MEMORY_CACHE_MAXSIZE = 64


@dataclass(frozen=True)
class _MemoryEntry:
    provider: str
    symbol: str
    stat_sig: tuple[int, int]
    result: MarketDataResult


class MarketDataCache:
    """Atomic JSON cache with bounded in-memory LRU and O(1) freshness index."""

    def __init__(
        self,
        directory: Path | str | None,
        *,
        enabled: bool = True,
        memory_maxsize: int = MEMORY_CACHE_MAXSIZE,
    ) -> None:
        self.directory = Path(directory) if directory else None
        self.enabled = bool(enabled and self.directory)
        self._memory_maxsize = max(1, int(memory_maxsize))
        self._lock = threading.RLock()
        self._memory_cache: OrderedDict[str, _MemoryEntry] = OrderedDict()
        self._freshness_index: dict[str, tuple[str, str, bool]] = {}
        self._freshness_by_session: dict[str, dict[tuple[str, bool], int]] = {}
        self._dir_mtime_ns: int | None = None
        if self.enabled and self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._sync_freshness_from_disk_locked()

    @staticmethod
    def _identity(provider: str, symbol: str) -> str:
        raw = f"{provider.lower()}:{symbol.upper()}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _path(self, provider: str, symbol: str) -> Path:
        if self.directory is None:
            raise RuntimeError("market-data cache has no directory")
        return self.directory / f"{self._identity(provider, symbol)}.json"

    @staticmethod
    def _shallow_copy_result(result: MarketDataResult) -> MarketDataResult:
        """Return a shallow frame copy with an isolated empty ``attrs`` dict."""
        frame_copy = result.frame.copy(deep=False)
        frame_copy.attrs = {}
        return replace(result, frame=frame_copy)

    def _index_set_locked(self, identity: str, data_as_of: str, provider: str, symbol: str) -> None:
        new_meta = (data_as_of, provider, symbol.upper().endswith(".L"))
        old_meta = self._freshness_index.get(identity)
        if old_meta == new_meta:
            return
        if old_meta is not None:
            self._index_decrement_locked(old_meta)
        self._freshness_index[identity] = new_meta
        session_buckets = self._freshness_by_session.setdefault(new_meta[0], {})
        bucket_key = (new_meta[1], new_meta[2])
        session_buckets[bucket_key] = session_buckets.get(bucket_key, 0) + 1

    def _index_remove_locked(self, identity: str) -> None:
        old_meta = self._freshness_index.pop(identity, None)
        if old_meta is not None:
            self._index_decrement_locked(old_meta)

    def _index_decrement_locked(self, meta: tuple[str, str, bool]) -> None:
        session_buckets = self._freshness_by_session.get(meta[0])
        if session_buckets is None:
            return
        bucket_key = (meta[1], meta[2])
        remaining = session_buckets.get(bucket_key, 0) - 1
        if remaining <= 0:
            session_buckets.pop(bucket_key, None)
            if not session_buckets:
                self._freshness_by_session.pop(meta[0], None)
        else:
            session_buckets[bucket_key] = remaining

    def _sync_freshness_from_disk_locked(self) -> None:
        if self.directory is None:
            return
        self._freshness_index.clear()
        self._freshness_by_session.clear()
        try:
            self._dir_mtime_ns = self.directory.stat().st_mtime_ns
        except OSError:
            self._dir_mtime_ns = None
            return
        for path in self.directory.glob("*.json"):
            try:
                if path.stat().st_size > MAX_CACHE_FILE_BYTES:
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
                    continue
                data_as_of = str(payload.get("data_as_of", ""))
                provider = str(payload.get("provider", ""))
                symbol = str(payload.get("symbol", "")).upper()
                if not data_as_of or not provider or not symbol:
                    continue
                self._index_set_locked(path.stem, data_as_of, provider, symbol)
            except (OSError, ValueError, TypeError):
                continue

    def _put_memory_locked(
        self,
        identity: str,
        *,
        provider: str,
        symbol: str,
        stat_sig: tuple[int, int],
        result: MarketDataResult,
    ) -> None:
        cached_frame = result.frame.loc[:, list(REQUIRED_OHLCV)].copy(deep=False)
        cached_frame.attrs = {}
        stored = MarketDataResult(
            frame=cached_frame,
            provider=result.provider,
            data_as_of=result.data_as_of,
            cache_status="hit",
            feed=result.feed,
            adjustment=result.adjustment,
            data_fingerprint=result.data_fingerprint,
            raw_source_fingerprint=result.raw_source_fingerprint,
            data_quality_audit=result.data_quality_audit,
            timestamp_kind="session_label",
            post_close_grace_used=False,
        )
        self._memory_cache[identity] = _MemoryEntry(
            provider=provider,
            symbol=symbol.upper(),
            stat_sig=stat_sig,
            result=stored,
        )
        self._memory_cache.move_to_end(identity)
        while len(self._memory_cache) > self._memory_maxsize:
            self._memory_cache.popitem(last=False)

    def load(
        self, provider: str, symbol: str, *, required_session: str, namespace: str | None = None
    ) -> MarketDataResult | None:
        if not self.enabled:
            return None
        identity = self._identity(namespace or provider, symbol)
        path = self._path(namespace or provider, symbol)
        upper_symbol = symbol.upper()
        with self._lock:
            try:
                st = path.stat()
                if st.st_size > MAX_CACHE_FILE_BYTES:
                    self._memory_cache.pop(identity, None)
                    return None
                stat_sig = (st.st_mtime_ns, st.st_size)
            except OSError:
                self._memory_cache.pop(identity, None)
                self._index_remove_locked(identity)
                return None

            mem_entry = self._memory_cache.get(identity)
            if mem_entry is not None and mem_entry.stat_sig == stat_sig:
                if (
                    mem_entry.provider == provider
                    and mem_entry.symbol == upper_symbol
                    and mem_entry.result.data_as_of == required_session
                ):
                    self._memory_cache.move_to_end(identity)
                    return self._shallow_copy_result(mem_entry.result)
                return None
            if mem_entry is not None:
                self._memory_cache.pop(identity, None)

            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("provider") != provider or payload.get("symbol") != upper_symbol:
                    return None
                if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
                    self._index_remove_locked(identity)
                    return None
                data_as_of = str(payload["data_as_of"])
                self._index_set_locked(identity, data_as_of, provider, upper_symbol)
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

            result = MarketDataResult(
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
            self._put_memory_locked(
                identity,
                provider=provider,
                symbol=upper_symbol,
                stat_sig=stat_sig,
                result=result,
            )
            return self._shallow_copy_result(result)

    def save(self, symbol: str, result: MarketDataResult, *, namespace: str | None = None) -> None:
        if not self.enabled:
            return
        identity = self._identity(namespace or result.provider, symbol)
        path = self._path(namespace or result.provider, symbol)
        upper_symbol = symbol.upper()
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "feed": result.feed,
            "adjustment": result.adjustment,
            "data_fingerprint": result.data_fingerprint,
            "raw_source_fingerprint": result.raw_source_fingerprint,
            "data_quality_audit": [record.to_dict() for record in result.data_quality_audit],
            "provider": result.provider,
            "symbol": upper_symbol,
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
            try:
                st = path.stat()
                stat_sig = (st.st_mtime_ns, st.st_size)
                if self.directory is not None:
                    self._dir_mtime_ns = self.directory.stat().st_mtime_ns
            except OSError:
                return
            self._index_set_locked(identity, result.data_as_of, result.provider, upper_symbol)
            self._put_memory_locked(
                identity,
                provider=result.provider,
                symbol=upper_symbol,
                stat_sig=stat_sig,
                result=result,
            )

    def fresh_entry_count(
        self,
        *,
        required_session: str,
        providers: set[str] | None = None,
        exclude_lse_symbols: bool = False,
    ) -> int:
        if not self.enabled or self.directory is None:
            return 0
        with self._lock:
            try:
                current_dir_mtime = self.directory.stat().st_mtime_ns
            except OSError:
                return 0
            if current_dir_mtime != self._dir_mtime_ns:
                self._sync_freshness_from_disk_locked()
            session_buckets = self._freshness_by_session.get(required_session)
            if not session_buckets:
                return 0
            count = 0
            for (provider, is_lse), bucket_count in session_buckets.items():
                if providers is not None and provider not in providers:
                    continue
                if exclude_lse_symbols and is_lse:
                    continue
                count += bucket_count
            return count
