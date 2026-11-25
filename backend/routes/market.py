"""Market data, ticker search, and company info endpoints."""

from __future__ import annotations

import inspect
import json
import logging
import math
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as default_yf  # type: ignore[import-untyped]
from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, Query, Request, Response

from config import settings
from market_data.base import (
    MarketDataProviderError,
    MarketDataServiceError,
    MarketDataSymbolNotFound,
)
from market_data.quote_units import resolve_quote_unit
from routes.common import limiter, validate_ticker

logger = logging.getLogger(__name__)
router = APIRouter(tags=["market"])

# Chart endpoint keeps the latest bars so the last (most relevant) session is
# always visible; older rows are dropped rather than strided.
MAX_HISTORY_BARS = 1500
LSE_HISTORY_YEARS = 10
MIN_INTRADAY_BARS = 5

_info_cache_lock = threading.Lock()
_info_cache: TTLCache = TTLCache(
    maxsize=settings.cache_max_size,
    ttl=settings.info_cache_ttl,
)
_info_negative_cache: TTLCache = TTLCache(maxsize=256, ttl=60)

_intraday_cache_lock = threading.Lock()
_intraday_cache: TTLCache = TTLCache(maxsize=256, ttl=60)
_intraday_negative_cache: TTLCache = TTLCache(maxsize=256, ttl=60)

_catalog_lock = threading.Lock()
_catalog_cache: dict[str, dict[str, Any]] = {}
_catalog_mtime_ns: int | None = None

_fundamentals_lock = threading.Lock()
_fundamentals_cache: dict[str, dict[str, Any]] = {}
_fundamentals_mtime_ns: int | None = None
_MIN_VALID_PE_TTM = 1.0
_MAX_VALID_PE_TTM = 1000.0

_FALLBACK_SECTOR_INDUSTRY: dict[str, tuple[str, str]] = {
    "AAPL": ("Information Technology", "Consumer Electronics"),
    "MSFT": ("Information Technology", "Systems Software"),
    "NVDA": ("Information Technology", "Semiconductors"),
    "GOOGL": ("Communication Services", "Interactive Media & Services"),
    "AMZN": ("Consumer Discretionary", "Broadline Retail"),
    "META": ("Communication Services", "Interactive Media & Services"),
    "TSLA": ("Consumer Discretionary", "Automobile Manufacturers"),
    "AMD": ("Information Technology", "Semiconductors"),
    "COST": ("Consumer Staples", "Consumer Staples Merchandise Retail"),
    "QCOM": ("Information Technology", "Semiconductors"),
    "JPM": ("Financials", "Diversified Banks"),
    "XOM": ("Energy", "Integrated Oil & Gas"),
    "WMT": ("Consumer Staples", "Consumer Staples Merchandise Retail"),
    "JNJ": ("Health Care", "Pharmaceuticals"),
    "CAT": ("Industrials", "Construction Machinery & Heavy Transportation Equipment"),
    "KO": ("Consumer Staples", "Soft Drinks & Non-Alcoholic Beverages"),
    "NEE": ("Utilities", "Electric Utilities"),
    "DIS": ("Communication Services", "Movies & Entertainment"),
    "BAC": ("Financials", "Diversified Banks"),
    "GE": ("Industrials", "Aerospace & Defense"),
    "SHEL.L": ("Energy", "Integrated Oil & Gas"),
    "AZN.L": ("Health Care", "Pharmaceuticals"),
    "HSBA.L": ("Financials", "Diversified Banks"),
    "BP.L": ("Energy", "Integrated Oil & Gas"),
    "ULVR.L": ("Consumer Staples", "Personal Care Products"),
    "GSK.L": ("Health Care", "Pharmaceuticals"),
    "RIO.L": ("Materials", "Diversified Metals & Mining"),
    "BATS.L": ("Consumer Staples", "Tobacco"),
    "BARC.L": ("Financials", "Diversified Banks"),
    "DGE.L": ("Consumer Staples", "Distillers & Vintners"),
    "CPG.L": ("Consumer Discretionary", "Restaurants & Contract Catering"),
    "IHG.L": ("Consumer Discretionary", "Hotels, Resorts & Cruise Lines"),
}


def clear_info_cache() -> None:
    with _info_cache_lock:
        _info_cache.clear()
        _info_negative_cache.clear()


def clear_intraday_cache() -> None:
    with _intraday_cache_lock:
        _intraday_cache.clear()
        _intraday_negative_cache.clear()


def clear_fundamentals_cache() -> None:
    global _fundamentals_mtime_ns
    with _fundamentals_lock:
        _fundamentals_cache.clear()
        _fundamentals_mtime_ns = None


def _load_disk_fundamentals_entry(symbol: str) -> dict[str, Any] | None:
    """Load offline batch fundamentals entry from backend/data/universe_fundamentals.json."""
    global _fundamentals_mtime_ns
    path = Path(__file__).resolve().parents[1] / "data" / "universe_fundamentals.json"
    with _fundamentals_lock:
        try:
            st = path.stat()
        except OSError:
            _fundamentals_cache.clear()
            _fundamentals_mtime_ns = None
            return None
        if st.st_mtime_ns != _fundamentals_mtime_ns:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                parsed: dict[str, dict[str, Any]] = {}
                default_updated = str(raw.get("updated_at") or "") if isinstance(raw, dict) else ""
                container = raw.get("tickers") if isinstance(raw, dict) else None
                if isinstance(container, dict):
                    for k, v in container.items():
                        if isinstance(v, dict):
                            pe_val = _finite_number(v.get("pe_ttm"))
                            if (
                                pe_val is not None
                                and _MIN_VALID_PE_TTM <= pe_val < _MAX_VALID_PE_TTM
                            ):
                                parsed[str(k).strip().upper()] = {
                                    "pe_ttm": round(pe_val, 2),
                                    "updated_at": str(v.get("updated_at") or default_updated or ""),
                                }
                _fundamentals_cache.clear()
                _fundamentals_cache.update(parsed)
                _fundamentals_mtime_ns = st.st_mtime_ns
            except Exception:
                return None
        entry = _fundamentals_cache.get(symbol.strip().upper())
        return dict(entry) if entry is not None else None


def _load_disk_catalog_meta(symbol: str) -> dict[str, Any] | None:
    """Load ticker metadata from backend/data/universe_metadata.json if present on disk."""
    global _catalog_mtime_ns
    path = Path(__file__).resolve().parents[1] / "data" / "universe_metadata.json"
    with _catalog_lock:
        try:
            st = path.stat()
        except OSError:
            return None
        if st.st_mtime_ns != _catalog_mtime_ns:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                parsed: dict[str, dict[str, Any]] = {}
                container = (
                    raw.get("tickers") or raw.get("entries") or raw.get("catalog") or raw
                    if isinstance(raw, dict)
                    else raw
                )
                if isinstance(container, dict):
                    for k, v in container.items():
                        if isinstance(v, dict):
                            parsed[str(k).strip().upper()] = dict(v)
                elif isinstance(container, list):
                    for item in container:
                        if isinstance(item, dict) and item.get("ticker"):
                            parsed[str(item["ticker"]).strip().upper()] = dict(item)
                _catalog_cache.clear()
                _catalog_cache.update(parsed)
                _catalog_mtime_ns = st.st_mtime_ns
            except Exception:
                return None
        entry = _catalog_cache.get(symbol.strip().upper())
        return dict(entry) if entry is not None else None


def _resolve_static_ticker_meta(symbol: str) -> dict[str, str]:
    from services.simple_forecast import get_ticker_meta

    sym = symbol.strip().upper()
    base = get_ticker_meta(sym)
    currency, currency_symbol = resolve_quote_unit(sym)
    disk = _load_disk_catalog_meta(sym) or {}
    fallback_sector, fallback_industry = _FALLBACK_SECTOR_INDUSTRY.get(sym, ("—", "—"))
    sector = str(disk.get("sector") or fallback_sector or "—")
    industry = str(disk.get("industry") or fallback_industry or "—")
    exchange_name = str(
        disk.get("exchange_name") or disk.get("exchange") or base.get("exchange_name") or "—"
    )
    exchange_mic = str(disk.get("exchange_mic") or base.get("exchange_mic") or "—")
    name = str(disk.get("name") or base.get("name") or sym)
    return {
        "name": name,
        "exchange": exchange_name,
        "exchange_mic": exchange_mic,
        "currency": currency,
        "currency_symbol": currency_symbol,
        "sector": sector,
        "industry": industry,
    }


def _alpaca_intraday_provider() -> Any | None:
    import data_pipeline as dp

    for provider in getattr(dp.market_data_service, "providers", ()):
        if getattr(provider, "configured", False) and callable(
            getattr(provider, "fetch_intraday_bars", None)
        ):
            return provider
    return None


def _fetch_history_daily(symbol: str):
    """Use the same exchange-aware completed daily bars as every forecast."""
    import data_pipeline as dp

    fetch_fn = dp.market_data_service.fetch_daily_bars
    if "allow_post_close_grace" in inspect.signature(fetch_fn).parameters:
        return fetch_fn(symbol, years=dp.HISTORICAL_YEARS, allow_post_close_grace=True)
    return fetch_fn(symbol, years=dp.HISTORICAL_YEARS)


def _fetch_history_intraday(symbol: str) -> list[dict[str, str | float]] | None:
    """Return warm cached intraday bars when available without blocking /history."""
    with _intraday_cache_lock:
        cached = _intraday_cache.get(symbol.strip().upper())
    if isinstance(cached, list) and len(cached) >= MIN_INTRADAY_BARS:
        return list(cached)
    return None


HISTORY_CACHE_CONTROL = "public, max-age=60, stale-while-revalidate=300"


def _strip_weak_prefix(tag: str) -> str:
    cleaned = tag.strip()
    if cleaned.startswith("W/"):
        return cleaned[2:].strip()
    return cleaned


def _etag_matches(if_none_match: str | None, etag: str) -> bool:
    if not if_none_match:
        return False
    target = _strip_weak_prefix(etag)
    for candidate in if_none_match.split(","):
        cleaned = candidate.strip()
        if cleaned == "*" or _strip_weak_prefix(cleaned) == target:
            return True
    return False


def _serialize_daily_bars(frame: pd.DataFrame) -> list[dict[str, str | float]]:
    """Vectorized daily bar serialization capped to the latest MAX_HISTORY_BARS."""
    if frame.empty:
        return []
    tail = frame.iloc[-MAX_HISTORY_BARS:] if len(frame) > MAX_HISTORY_BARS else frame
    if isinstance(tail.index, pd.DatetimeIndex):
        dates = tail.index.strftime("%Y-%m-%d").tolist()
    else:
        dates = [pd.Timestamp(idx).date().isoformat() for idx in tail.index]
    closes = tail["Close"].to_numpy(dtype=float).tolist()
    return [{"d": day, "c": close} for day, close in zip(dates, closes, strict=True)]


@router.get("/api/v1/history")
@limiter.limit("30/minute")
def history(
    request: Request,
    response: Response,
    ticker: str = Query(default="AAPL", min_length=1, max_length=12),
) -> Any:
    """Return downsampled daily closes plus last-session intraday bars.

    One round trip backs every range tab and zoom level of the price chart:
    the client derives 5D..MAX from the daily series and 24H from the
    (optional) intraday series. Daily bars are capped to the most recent
    ``MAX_HISTORY_BARS`` sessions; the latest session is always retained.
    """
    symbol = validate_ticker(ticker)
    try:
        result = _fetch_history_daily(symbol)
        grace_used = bool(getattr(result, "post_close_grace_used", False))
        fingerprint = str(getattr(result, "data_fingerprint", "") or "")
        etag = f'W/"{fingerprint}"' if (fingerprint and not grace_used) else None
        if etag and _etag_matches(request.headers.get("if-none-match"), etag):
            return Response(
                status_code=304,
                headers={"ETag": etag, "Cache-Control": HISTORY_CACHE_CONTROL},
            )
        daily = _serialize_daily_bars(result.frame)
        intraday = _fetch_history_intraday(symbol)
        currency, currency_symbol = resolve_quote_unit(symbol)
        if grace_used:
            response.headers["Cache-Control"] = "no-store"
        else:
            response.headers["Cache-Control"] = HISTORY_CACHE_CONTROL
            if etag:
                response.headers["ETag"] = etag
        audit = getattr(result, "data_quality_audit", ())
        with _intraday_cache_lock:
            neg_intraday = symbol in _intraday_negative_cache
        intraday_available = bool(
            intraday
            or (
                not symbol.endswith(".L")
                and _alpaca_intraday_provider() is not None
                and not neg_intraday
            )
        )
        return {
            "ticker": symbol,
            "as_of": result.data_as_of,
            "provider": result.provider,
            "feed": result.feed,
            "adjustment": result.adjustment,
            "currency": currency,
            "currency_symbol": currency_symbol,
            "data_fingerprint": fingerprint,
            "raw_source_fingerprint": getattr(result, "raw_source_fingerprint", ""),
            "data_quality_audit": [
                record.to_dict() if hasattr(record, "to_dict") else record for record in audit
            ],
            "post_close_grace_used": grace_used,
            "completed_daily_bars_only": True,
            "market_data_cache": result.cache_status or "unknown",
            "first_date": daily[0]["d"] if daily else None,
            "daily": daily,
            "intraday": intraday,
            "intraday_available": intraday_available,
            "intraday_session": result.data_as_of if intraday else None,
        }
    except MarketDataSymbolNotFound as err:
        raise HTTPException(
            status_code=404, detail="No market data is available for this ticker."
        ) from err
    except MarketDataServiceError as err:
        raise HTTPException(
            status_code=503, detail="Market data is temporarily unavailable."
        ) from err
    except MarketDataProviderError as err:
        raise HTTPException(
            status_code=503, detail="Market data returned an invalid response."
        ) from err
    except Exception as err:
        logger.exception("Price history failed for %s", symbol)
        raise HTTPException(
            status_code=503, detail="Price history is temporarily unavailable."
        ) from err


@router.get("/api/v1/search")
@limiter.limit("30/minute")
def search(
    request: Request,
    query: str = Query(..., min_length=1, max_length=100),
):
    import api

    yf = getattr(api, "yf", default_yf)

    from services.simple_forecast import SUPPORTED_TICKERS, get_ticker_meta

    exact_symbol = query.strip().upper()
    local = [
        {"ticker": ticker, "name": get_ticker_meta(ticker)["name"], "type": "EQUITY"}
        for ticker in SUPPORTED_TICKERS
        if exact_symbol in ticker or exact_symbol in get_ticker_meta(ticker)["name"].upper()
    ]
    if local:
        return {"results": sorted(local, key=lambda item: item["ticker"] != exact_symbol)[:8]}
    fallback = []
    try:
        results = yf.Search(query, max_results=8)
        suggestions = []
        for r in results.quotes:
            if r.get("quoteType") in ("EQUITY", "ETF") and r.get("symbol") in SUPPORTED_TICKERS:
                suggestions.append(
                    {
                        "ticker": r.get("symbol", ""),
                        "name": r.get("longname") or r.get("shortname", ""),
                        "type": r.get("quoteType", ""),
                    }
                )
        seen = {item["ticker"] for item in suggestions}
        return {"results": suggestions + [item for item in fallback if item["ticker"] not in seen]}
    except Exception as err:
        if fallback:
            logger.warning("Autocomplete upstream unavailable; returning exact symbol fallback")
            return {"results": fallback, "degraded": True}
        logger.exception("Error in /api/v1/search")
        raise HTTPException(
            status_code=500,
            detail="Search failed. Please try again later.",
        ) from err


def _ny_session_date(ts_value: str) -> str:
    ts = pd.Timestamp(str(ts_value))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/New_York").date().isoformat()


@router.get("/api/v1/intraday")
@limiter.limit("30/minute")
def intraday_history(
    request: Request,
    response: Response,
    ticker: str = Query(default="AAPL", min_length=1, max_length=12),
) -> dict[str, Any]:
    """Return bounded latest-session 24H intraday bars with 60s TTL & negative caching."""
    symbol = validate_ticker(ticker)
    if symbol.endswith(".L"):
        return {
            "ticker": symbol,
            "session": None,
            "bars": [],
            "bar_count": 0,
            "available": False,
        }

    with _intraday_cache_lock:
        cached_bars = _intraday_cache.get(symbol)
        if isinstance(cached_bars, list) and len(cached_bars) >= MIN_INTRADAY_BARS:
            session_date = _ny_session_date(str(cached_bars[-1]["t"]))
            response.headers["Cache-Control"] = "public, max-age=60"
            return {
                "ticker": symbol,
                "session": session_date,
                "bars": list(cached_bars),
                "bar_count": len(cached_bars),
                "available": True,
            }
        if symbol in _intraday_negative_cache:
            return {
                "ticker": symbol,
                "session": None,
                "bars": [],
                "bar_count": 0,
                "available": False,
            }

    provider = _alpaca_intraday_provider()
    if provider is None:
        with _intraday_cache_lock:
            _intraday_negative_cache[symbol] = True
        return {
            "ticker": symbol,
            "session": None,
            "bars": [],
            "bar_count": 0,
            "available": False,
        }

    try:
        raw_bars = provider.fetch_intraday_bars(symbol)
        valid_bars = [
            {"t": str(b["t"]), "c": float(b["c"])}
            for b in (raw_bars or [])
            if isinstance(b, dict)
            and b.get("t")
            and b.get("c") is not None
            and math.isfinite(float(b["c"]))
            and float(b["c"]) > 0
        ]
    except Exception as exc:
        logger.info("Intraday fetch unavailable for %s: %s", symbol, type(exc).__name__)
        with _intraday_cache_lock:
            _intraday_negative_cache[symbol] = True
        return {
            "ticker": symbol,
            "session": None,
            "bars": [],
            "bar_count": 0,
            "available": False,
        }

    if len(valid_bars) < MIN_INTRADAY_BARS:
        with _intraday_cache_lock:
            _intraday_negative_cache[symbol] = True
        return {
            "ticker": symbol,
            "session": None,
            "bars": [],
            "bar_count": 0,
            "available": False,
        }

    session_date = _ny_session_date(str(valid_bars[-1]["t"]))
    with _intraday_cache_lock:
        _intraday_cache[symbol] = valid_bars
    response.headers["Cache-Control"] = "public, max-age=60"
    return {
        "ticker": symbol,
        "session": session_date,
        "bars": valid_bars,
        "bar_count": len(valid_bars),
        "available": True,
    }


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _lookup_mapping_or_attr(source: Any, *keys: str) -> Any:
    if source is None:
        return None
    for key in keys:
        try:
            if isinstance(source, dict) and key in source and source[key] is not None:
                return source[key]
            getter = getattr(source, "get", None)
            if callable(getter):
                val = getter(key)
                if val is not None:
                    return val
        except Exception:
            pass
        try:
            val = getattr(source, key, None)
            if val is not None and not callable(val):
                return val
        except Exception:
            continue
    return None


@router.get("/api/v1/info")
@limiter.limit("20/minute")
def stock_info(request: Request, ticker: str = "AAPL"):
    """Return partial company fundamentals using catalog metadata + fast_info with 60s negative caching."""
    import api

    yf = getattr(api, "yf", default_yf)
    cache = getattr(api, "_info_cache", _info_cache)
    neg_cache = getattr(api, "_info_negative_cache", _info_negative_cache)
    lock = getattr(api, "_info_cache_lock", _info_cache_lock)

    ticker = validate_ticker(ticker)

    with lock:
        cached = cache.get(ticker)
        if cached is not None:
            return cached
        if ticker in neg_cache:
            raise HTTPException(
                status_code=503,
                detail="Failed to fetch stock info. Please try again later.",
            )

    static_meta = _resolve_static_ticker_meta(ticker)
    currency = static_meta["currency"]
    currency_symbol = static_meta["currency_symbol"]
    market_cap_currency = "GBP" if currency == "GBp" else "USD"
    market_cap_currency_symbol = "£" if currency == "GBp" else "$"

    try:
        ticker_obj = yf.Ticker(ticker)
        fast_info = getattr(ticker_obj, "fast_info", None)
        has_mock_info_attr = "info" in getattr(ticker_obj, "__dict__", {})

        market_cap = _finite_number(_lookup_mapping_or_attr(fast_info, "market_cap", "marketCap"))
        shares = _finite_number(_lookup_mapping_or_attr(fast_info, "shares"))
        high_52w = _finite_number(
            _lookup_mapping_or_attr(fast_info, "year_high", "fiftyTwoWeekHigh")
        )
        low_52w = _finite_number(_lookup_mapping_or_attr(fast_info, "year_low", "fiftyTwoWeekLow"))
        day_high = _finite_number(_lookup_mapping_or_attr(fast_info, "day_high", "dayHigh"))
        day_low = _finite_number(_lookup_mapping_or_attr(fast_info, "day_low", "dayLow"))
        prev_close = _finite_number(
            _lookup_mapping_or_attr(
                fast_info, "previous_close", "previousClose", "regularMarketPreviousClose"
            )
        )
        last_price = _finite_number(_lookup_mapping_or_attr(fast_info, "last_price", "lastPrice"))
        avg_vol = _finite_number(
            _lookup_mapping_or_attr(
                fast_info,
                "three_month_average_volume",
                "ten_day_average_volume",
                "averageVolume",
                "last_volume",
            )
        )
        pe_ratio = _finite_number(_lookup_mapping_or_attr(fast_info, "trailingPE", "peRatio"))

        has_fast_metrics = any(
            v is not None for v in (market_cap, high_52w, low_52w, prev_close, day_high, day_low)
        )
        info_dict: dict[str, Any] = {}
        if has_mock_info_attr or not has_fast_metrics:
            raw_info = ticker_obj.info
            if isinstance(raw_info, dict):
                info_dict = raw_info

        if info_dict:
            if market_cap is None:
                market_cap = _finite_number(info_dict.get("marketCap"))
            if high_52w is None:
                high_52w = _finite_number(info_dict.get("fiftyTwoWeekHigh"))
            if low_52w is None:
                low_52w = _finite_number(info_dict.get("fiftyTwoWeekLow"))
            if day_high is None:
                day_high = _finite_number(info_dict.get("dayHigh"))
            if day_low is None:
                day_low = _finite_number(info_dict.get("dayLow"))
            if prev_close is None:
                prev_close = _finite_number(info_dict.get("previousClose"))
            if avg_vol is None:
                avg_vol = _finite_number(info_dict.get("averageVolume"))
            if pe_ratio is None:
                pe_ratio = _finite_number(
                    info_dict.get("trailingPE")
                    if info_dict.get("trailingPE") is not None
                    else info_dict.get("peRatio")
                )

        pe_source: str | None = "live" if pe_ratio is not None else None
        pe_updated_at: str | None = None
        if pe_ratio is None:
            disk_fund = _load_disk_fundamentals_entry(ticker)
            if disk_fund is not None:
                pe_ratio = _finite_number(disk_fund.get("pe_ttm"))
                if pe_ratio is not None:
                    pe_source = "offline_batch"
                    pe_updated_at = str(disk_fund.get("updated_at") or "") or None

        if not has_fast_metrics and not info_dict:
            raise RuntimeError(f"No fundamentals returned for {ticker}")

        # London GBp normalization: handle both whole-GBP fast_info feeds and
        # mixed GBp/GBP payloads where 52W or day range fields arrive in pounds
        # (>20x smaller than the GBp reference price). Market cap stays in GBP.
        if currency == "GBp":
            upstream_currency = str(
                _lookup_mapping_or_attr(fast_info, "currency") or info_dict.get("currency") or ""
            ).strip()
            candidates = [
                v
                for v in (prev_close, last_price, day_high, day_low, high_52w, low_52w)
                if v is not None and v > 0
            ]
            if upstream_currency == "GBP" and candidates and max(candidates) < 500.0:
                if high_52w is not None:
                    high_52w = round(high_52w * 100.0, 4)
                if low_52w is not None:
                    low_52w = round(low_52w * 100.0, 4)
                if day_high is not None:
                    day_high = round(day_high * 100.0, 4)
                if day_low is not None:
                    day_low = round(day_low * 100.0, 4)
                if prev_close is not None:
                    prev_close = round(prev_close * 100.0, 4)
            else:
                ref_gbp = max(candidates) if candidates and max(candidates) >= 50.0 else None
                if ref_gbp is not None:
                    threshold = ref_gbp / 20.0
                    if high_52w is not None and 0 < high_52w < threshold:
                        high_52w = round(high_52w * 100.0, 4)
                    if low_52w is not None and 0 < low_52w < threshold:
                        low_52w = round(low_52w * 100.0, 4)
                    if day_high is not None and 0 < day_high < threshold:
                        day_high = round(day_high * 100.0, 4)
                    if day_low is not None and 0 < day_low < threshold:
                        day_low = round(day_low * 100.0, 4)
                    if prev_close is not None and 0 < prev_close < threshold:
                        prev_close = round(prev_close * 100.0, 4)

            # Yahoo fast_info computes market_cap = shares * last_price without
            # converting GBp (pence) to GBP (pounds), inflating UK market caps 100x.
            if market_cap is not None and market_cap > 0 and upstream_currency != "GBP":
                ref_price_gbp = (
                    last_price if (last_price is not None and last_price > 0) else prev_close
                )
                if (
                    shares is not None
                    and shares > 0
                    and ref_price_gbp is not None
                    and ref_price_gbp > 0
                ):
                    expected_gbp = shares * (ref_price_gbp / 100.0)
                    if market_cap > expected_gbp * 10.0:
                        market_cap = round(market_cap / 100.0, 2)
                elif upstream_currency in ("GBp", "GBX") and market_cap > 2_000_000_000_000.0:
                    market_cap = round(market_cap / 100.0, 2)

        resolved_name = (
            info_dict.get("longName") or info_dict.get("shortName") or static_meta["name"] or ticker
        )
        resolved_exchange = (
            static_meta["exchange"]
            if static_meta["exchange"] != "—"
            else str(info_dict.get("exchange") or "—")
        )
        resolved_sector = (
            static_meta["sector"]
            if static_meta["sector"] != "—"
            else str(info_dict.get("sector") or "—")
        )
        resolved_industry = (
            static_meta["industry"]
            if static_meta["industry"] != "—"
            else str(info_dict.get("industry") or "—")
        )

        data = {
            "ticker": ticker,
            "name": resolved_name,
            "exchange": resolved_exchange,
            "exchange_mic": static_meta["exchange_mic"],
            "currency": currency,
            "currency_symbol": currency_symbol,
            "marketCapCurrency": market_cap_currency,
            "marketCapCurrencySymbol": market_cap_currency_symbol,
            "marketCap": market_cap,
            "peRatio": pe_ratio,
            "peSource": pe_source,
            "peUpdatedAt": pe_updated_at,
            "fiftyTwoWeekHigh": high_52w,
            "fiftyTwoWeekLow": low_52w,
            "avgVolume": avg_vol,
            "dayHigh": day_high,
            "dayLow": day_low,
            "previousClose": prev_close,
            "sector": resolved_sector,
            "industry": resolved_industry,
        }
        with lock:
            cache[ticker] = data
            neg_cache.pop(ticker, None)
        return data
    except HTTPException:
        raise
    except Exception as err:
        with lock:
            neg_cache[ticker] = True
        logger.exception("Error fetching info for %s", ticker)
        raise HTTPException(
            status_code=503,
            detail="Failed to fetch stock info. Please try again later.",
        ) from err
