"""Provider-neutral market-data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class MarketDataProviderError(RuntimeError):
    """Base class for provider failures."""


class MarketDataServiceError(MarketDataProviderError):
    """The provider is temporarily unable to serve data."""


class MarketDataRateLimitError(MarketDataServiceError):
    """The provider rejected the request because a quota was exceeded."""


class MarketDataTimeoutError(MarketDataServiceError):
    """The provider did not answer within the configured deadline."""


class MarketDataConfigurationError(MarketDataServiceError):
    """Required server-side provider configuration is absent or invalid."""


class MarketDataSymbolNotFound(MarketDataProviderError):
    """A provider authoritatively reported that a symbol has no data."""


@dataclass(frozen=True)
class BarRepairRecord:
    """Immutable audit record of a single conservative OHLCV repair or empty-session drop."""

    date: str
    field: str
    original: float | None
    repaired: float | None
    reason: str

    def to_dict(self) -> dict[str, str | float | None]:
        return {
            "date": self.date,
            "field": self.field,
            "original": self.original,
            "repaired": self.repaired,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BarRepairRecord:
        orig = payload.get("original")
        rep = payload.get("repaired")
        return cls(
            date=str(payload["date"]),
            field=str(payload["field"]),
            original=None if orig is None else float(orig),  # type: ignore[arg-type]
            repaired=None if rep is None else float(rep),  # type: ignore[arg-type]
            reason=str(payload["reason"]),
        )


@dataclass(frozen=True)
class MarketDataResult:
    """Normalized daily OHLCV bars and their acquisition provenance."""

    frame: pd.DataFrame
    provider: str
    data_as_of: str
    cache_status: str = "miss"
    feed: str = "unknown"
    adjustment: str = "unknown"
    data_fingerprint: str = ""
    raw_source_fingerprint: str = ""
    data_quality_audit: tuple[BarRepairRecord, ...] = ()
    timestamp_kind: str = "session_label"
    post_close_grace_used: bool = False


class MarketDataProvider(Protocol):
    """Interface implemented by daily-bar providers."""

    name: str

    @property
    def configured(self) -> bool: ...

    def fetch_daily_bars(self, symbol: str, *, years: int) -> MarketDataResult: ...
