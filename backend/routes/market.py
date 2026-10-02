"""Market data, ticker search, and company info endpoints."""

from __future__ import annotations

import logging
import threading
from typing import Any

import yfinance as default_yf  # type: ignore[import-untyped]
from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, Query, Request

from config import settings
from market_data.base import (
    MarketDataProviderError,
    MarketDataServiceError,
    MarketDataSymbolNotFound,
)
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

    return dp.market_data_service.fetch_daily_bars(symbol, years=dp.HISTORICAL_YEARS)


def _fetch_history_intraday(symbol: str) -> list[dict[str, str | float]] | None:
    """Intraday bars for the 24H tab when a live session source is available.

    Returns ``None`` until a live intraday provider is configured; the chart
    hides the 24H tab when intraday is absent (see priceRanges.js).
    """
    return None


@router.get("/api/v1/history")
@limiter.limit("30/minute")
def history(
    request: Request,
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
        frame = result.frame
        daily = [
            {"d": index.date().isoformat(), "c": float(row["Close"])}
            for index, row in frame.iterrows()
        ]
        if len(daily) > MAX_HISTORY_BARS:
            daily = daily[-MAX_HISTORY_BARS:]
        intraday = _fetch_history_intraday(symbol)
        return {
            "ticker": symbol,
            "as_of": result.data_as_of,
            "provider": result.provider,
            "feed": result.feed,
            "adjustment": result.adjustment,
            "data_fingerprint": result.data_fingerprint,
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
