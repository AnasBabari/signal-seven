"""Unified 286-ticker tri-exchange metadata catalog lookup service."""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
from typing import Any

from market_data.quote_units import LSE_QUOTE_UNIT_OVERRIDES

_CATALOG_CANDIDATES: tuple[Path, ...] = (
    Path(__file__).resolve().parents[1] / "data" / "universe_metadata.json",
    Path(__file__).resolve().parents[2] / "backend" / "data" / "universe_metadata.json",
)


@functools.lru_cache(maxsize=1)
def _load_cached_catalog() -> dict[str, Any]:
    for path in _CATALOG_CANDIDATES:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("tickers"), dict):
                return data
    raise FileNotFoundError("backend/data/universe_metadata.json is missing or invalid")


def clear_universe_catalog_cache() -> None:
    """Clear the in-memory catalog cache (used in tests)."""
    _load_cached_catalog.cache_clear()


def load_universe_catalog() -> dict[str, Any]:
    """Return a copy of the full 286-ticker universe metadata catalog payload."""
    return copy.deepcopy(_load_cached_catalog())


def normalize_sector(raw_sector: str | None) -> str:
    """Normalize a raw provider sector label into a canonical 11-sector GICS label."""
    if not raw_sector:
        return "Unknown"
    cleaned = str(raw_sector).strip()
    if not cleaned or cleaned in {"—", "-", "N/A", "Unknown"}:
        return "Unknown"
    crosswalk = _load_cached_catalog().get("sector_crosswalk", {})
    return str(crosswalk.get(cleaned, cleaned))


def get_catalog_ticker_meta(symbol: str) -> dict[str, Any]:
    """Return rich catalog metadata for ``symbol`` (with M2 LSE quote-unit overrides)."""
    sym = str(symbol or "").strip().upper()
    catalog = _load_cached_catalog()
    tickers: dict[str, dict[str, Any]] = catalog["tickers"]
    override = LSE_QUOTE_UNIT_OVERRIDES.get(sym)

    if sym in tickers:
        meta = dict(tickers[sym])
        if override is not None:
            meta["currency"] = override["currency"]
            meta["currency_symbol"] = override["currency_symbol"]
        return meta

    if override is not None:
        return {
            "ticker": sym,
            "name": override["name"],
            "exchange": override["exchange_name"],
            "exchange_name": override["exchange_name"],
            "exchange_mic": override["exchange_mic"],
            "currency": override["currency"],
            "currency_symbol": override["currency_symbol"],
            "sector": "Unknown",
            "industry": "Unknown",
            "cik": None,
        }

    if sym.endswith(".L"):
        return {
            "ticker": sym,
            "name": sym,
            "exchange": "LSE",
            "exchange_name": "LSE",
            "exchange_mic": "XLON",
            "currency": "GBp",
            "currency_symbol": "p",
            "sector": "Unknown",
            "industry": "Unknown",
            "cik": None,
        }

    return {
        "ticker": sym,
        "name": sym,
        "exchange": "NASDAQ",
        "exchange_name": "NASDAQ",
        "exchange_mic": "XNAS",
        "currency": "USD",
        "currency_symbol": "$",
        "sector": "Unknown",
        "industry": "Unknown",
        "cik": None,
    }


def get_ticker_cik(symbol: str) -> str | None:
    """Return the 10-digit zero-padded SEC CIK for a US ticker, or ``None`` for LSE/unknown."""
    sym = str(symbol or "").strip().upper()
    if not sym or sym.endswith(".L"):
        return None
    catalog = _load_cached_catalog()
    entry = catalog["tickers"].get(sym)
    if entry is None:
        return None
    cik = entry.get("cik")
    return str(cik) if isinstance(cik, str) and cik else None


__all__ = [
    "clear_universe_catalog_cache",
    "get_catalog_ticker_meta",
    "get_ticker_cik",
    "load_universe_catalog",
    "normalize_sector",
]
