"""Market data, ticker search, and company info endpoints."""

from __future__ import annotations

import inspect
import logging
import threading
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

_info_cache_lock = threading.Lock()
_info_cache: TTLCache = TTLCache(
    maxsize=settings.cache_max_size,
    ttl=settings.info_cache_ttl,
)


def _fetch_history_daily(symbol: str):
    """Use the same exchange-aware completed daily bars as every forecast."""
    import data_pipeline as dp

    fetch_fn = dp.market_data_service.fetch_daily_bars
    if "allow_post_close_grace" in inspect.signature(fetch_fn).parameters:
        return fetch_fn(symbol, years=dp.HISTORICAL_YEARS, allow_post_close_grace=True)
    return fetch_fn(symbol, years=dp.HISTORICAL_YEARS)


def _fetch_history_intraday(symbol: str) -> list[dict[str, str | float]] | None:
    """Intraday bars for the 24H tab when a live session source is available.

    Returns ``None`` until a live intraday provider is configured; the chart
    hides the 24H tab when intraday is absent (see priceRanges.js).
    """
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


@router.get("/api/v1/info")
@limiter.limit("20/minute")
def stock_info(request: Request, ticker: str = "AAPL"):
    """Return rich metadata for a ticker."""
    import api

    yf = getattr(api, "yf", default_yf)
    cache = getattr(api, "_info_cache", _info_cache)
    lock = getattr(api, "_info_cache_lock", _info_cache_lock)

    ticker = validate_ticker(ticker)

    with lock:
        cached = cache.get(ticker)
    if cached:
        return cached

    try:
        info = yf.Ticker(ticker).info
        data = {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName", ticker),
            "exchange": info.get("exchange", "—"),
            "currency": info.get("currency", "USD"),
            "marketCap": info.get("marketCap"),
            "peRatio": info.get("trailingPE"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "avgVolume": info.get("averageVolume"),
            "dayHigh": info.get("dayHigh"),
            "dayLow": info.get("dayLow"),
            "previousClose": info.get("previousClose"),
            "sector": info.get("sector", "—"),
            "industry": info.get("industry", "—"),
        }
        with lock:
            cache[ticker] = data
        return data
    except Exception as err:
        logger.exception("Error fetching info for %s", ticker)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch stock info. Please try again later.",
        ) from err
