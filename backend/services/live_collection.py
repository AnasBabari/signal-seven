"""Frozen operational contract for genuine forward-volatility collection."""

from __future__ import annotations

from datetime import date
from itertools import product
from types import MappingProxyType

from services.volatility_contract import SUPPORTED_VOLATILITY_HORIZONS

LIVE_UNIVERSE_V1 = (
    "MSFT",
    "AAPL",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "JPM",
    "XOM",
    "JNJ",
    "WMT",
    "CAT",
    "NEE",
    "PLD",
    "KO",
    "NMM",
)
LIVE_HORIZONS_V1 = SUPPORTED_VOLATILITY_HORIZONS
LIVE_UNIVERSE_VERSION = "live_universe_v1"
LIVE_START_DATE = date.fromisoformat("2026-09-02")

# Operational V1 live-collection policy is frozen independently of public
# VOLATILITY_DEFAULT_POLICY so public routing experiments never break V1 ledger continuity.
LIVE_MODEL_POLICY_V1 = MappingProxyType(
    {
        1: "garch_11",
        5: "rolling_mean",
        10: "rolling_mean",
        20: "rolling_mean",
    }
)
LIVE_COLLECTION_ITEMS_V1 = tuple(product(LIVE_UNIVERSE_V1, LIVE_HORIZONS_V1))
LIVE_EXPECTED_RECORD_COUNT = len(LIVE_COLLECTION_ITEMS_V1)

# Versioned V2 cohort (20 US + 10 LSE) resolving the V1 PLD/NMM boundary cleanly
# while leaving LIVE_UNIVERSE_V1 history and continuity untouched.
LIVE_UNIVERSE_VERSION_V2 = "live_v2_2026_11"
LIVE_UNIVERSE_V2_US = (
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
    "COST",
    "QCOM",
    "JPM",
    "XOM",
    "WMT",
    "JNJ",
    "CAT",
    "KO",
    "NEE",
    "DIS",
    "BAC",
    "GE",
)
LIVE_UNIVERSE_V2_LSE = (
    "SHEL.L",
    "AZN.L",
    "HSBA.L",
    "BP.L",
    "ULVR.L",
    "GSK.L",
    "RIO.L",
    "BATS.L",
    "BARC.L",
    "DGE.L",
)
LIVE_UNIVERSE_V2 = LIVE_UNIVERSE_V2_US + LIVE_UNIVERSE_V2_LSE
LIVE_HORIZONS_V2 = SUPPORTED_VOLATILITY_HORIZONS
LIVE_MODEL_POLICY_V2 = LIVE_MODEL_POLICY_V1
LIVE_COLLECTION_ITEMS_V2 = tuple(product(LIVE_UNIVERSE_V2, LIVE_HORIZONS_V2))


def exchange_for_live_ticker(ticker: str) -> str:
    """Return the collection exchange group ('LSE' or 'NYSE') for a ticker."""
    return "LSE" if str(ticker).strip().upper().endswith(".L") else "NYSE"


def resolve_live_cohort_version(cohort: str = "v1") -> str:
    key = str(cohort).strip().lower()
    if key in ("v1", LIVE_UNIVERSE_VERSION):
        return LIVE_UNIVERSE_VERSION
    if key in ("v2", LIVE_UNIVERSE_VERSION_V2):
        return LIVE_UNIVERSE_VERSION_V2
    raise ValueError(f"Unsupported live collection cohort: {cohort!r}")


def resolve_live_cohort_items(
    cohort: str = "v1",
    *,
    exchange: str = "ALL",
    batch_size: int | None = None,
    batch_index: int = 0,
) -> tuple[tuple[str, int], ...]:
    """Return deterministic (ticker, horizon) items for a cohort, exchange, and batch slice."""
    version = resolve_live_cohort_version(cohort)
    tickers = LIVE_UNIVERSE_V1 if version == LIVE_UNIVERSE_VERSION else LIVE_UNIVERSE_V2
    horizons = LIVE_HORIZONS_V1 if version == LIVE_UNIVERSE_VERSION else LIVE_HORIZONS_V2

    ex = str(exchange).strip().upper()
    if ex not in ("ALL", "NYSE", "LSE"):
        raise ValueError(f"Unsupported collection exchange filter: {exchange!r}")
    if ex != "ALL":
        tickers = tuple(t for t in tickers if exchange_for_live_ticker(t) == ex)

    items = tuple(product(tickers, horizons))
    if batch_size is not None:
        size = int(batch_size)
        idx = int(batch_index)
        if size < 1:
            raise ValueError("batch_size must be >= 1")
        if idx < 0:
            raise ValueError("batch_index must be >= 0")
        start = idx * size
        items = items[start : start + size]
    return items


def validate_live_collection_item(
    ticker: str, horizon: int, *, cohort: str = "v1"
) -> tuple[str, int]:
    """Return a normalized frozen-universe item or fail closed."""
    version = resolve_live_cohort_version(cohort)
    universe = LIVE_UNIVERSE_V1 if version == LIVE_UNIVERSE_VERSION else LIVE_UNIVERSE_V2
    horizons = LIVE_HORIZONS_V1 if version == LIVE_UNIVERSE_VERSION else LIVE_HORIZONS_V2
    symbol = str(ticker).strip().upper()
    normalized_horizon = int(horizon)
    if symbol not in universe:
        raise ValueError("Ticker is not part of the frozen live universe.")
    if normalized_horizon not in horizons:
        raise ValueError("Horizon is not part of the frozen live contract.")
    return symbol, normalized_horizon


__all__ = [
    "LIVE_COLLECTION_ITEMS_V1",
    "LIVE_COLLECTION_ITEMS_V2",
    "LIVE_EXPECTED_RECORD_COUNT",
    "LIVE_HORIZONS_V1",
    "LIVE_HORIZONS_V2",
    "LIVE_MODEL_POLICY_V1",
    "LIVE_MODEL_POLICY_V2",
    "LIVE_START_DATE",
    "LIVE_UNIVERSE_V1",
    "LIVE_UNIVERSE_V2",
    "LIVE_UNIVERSE_V2_LSE",
    "LIVE_UNIVERSE_V2_US",
    "LIVE_UNIVERSE_VERSION",
    "LIVE_UNIVERSE_VERSION_V2",
    "exchange_for_live_ticker",
    "resolve_live_cohort_items",
    "resolve_live_cohort_version",
    "validate_live_collection_item",
]
