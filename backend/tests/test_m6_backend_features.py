"""Milestone 06 (PRs 6B, 6C, 6D) backend feature and reliability tests."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
from api import app
from market_data.alpaca import AlpacaProvider
from market_data.base import MarketDataResult
from routes import market as market_route
from routes import volatility as volatility_route
from routes.common import limiter
from scripts import collect_live_forecasts as collector_script
from scripts.collect_live_forecasts import (
    ApiResult,
    collection_session,
    parse_args,
    run_live_collection,
    run_preflight,
    run_settlement,
)
from services import market_news
from services.forecast_ledger import ForecastLedger
from services.live_collection import (
    LIVE_COLLECTION_ITEMS_V1,
    LIVE_COLLECTION_ITEMS_V2,
    LIVE_EXPECTED_RECORD_COUNT,
    LIVE_MODEL_POLICY_V1,
    LIVE_UNIVERSE_V1,
    LIVE_UNIVERSE_V2,
    LIVE_UNIVERSE_V2_LSE,
    LIVE_UNIVERSE_V2_US,
    LIVE_UNIVERSE_VERSION,
    LIVE_UNIVERSE_VERSION_V2,
    resolve_live_cohort_items,
    validate_live_collection_item,
)
from services.volatility_snapshot import VOLATILITY_HORIZONS

CLIENT = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_m6_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(limiter, "enabled", False)
    market_route.clear_info_cache()
    market_route.clear_intraday_cache()
    market_news.clear_news_cache()
    yield
    market_route.clear_info_cache()
    market_route.clear_intraday_cache()
    market_news.clear_news_cache()


# ---------------------------------------------------------------------------
# PR 6B: Fundamentals (/api/v1/info)
# ---------------------------------------------------------------------------


def test_info_uses_fast_info_without_touching_slow_info_and_nullable_pe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"ticker": 0}

    class _FastOnlyTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol
            self.fast_info = {
                "market_cap": 3_150_000_000_000.0,
                "year_high": 242.5,
                "year_low": 168.0,
                "day_high": 231.0,
                "day_low": 227.5,
                "previous_close": 229.0,
                "three_month_average_volume": 48_000_000.0,
                "currency": "USD",
            }

        @property
        def info(self) -> dict[str, Any]:
            raise AssertionError("slow .info must not be touched when fast_info succeeds")

    class _FakeYF:
        def Ticker(self, symbol: str) -> _FastOnlyTicker:
            calls["ticker"] += 1
            return _FastOnlyTicker(symbol)

    monkeypatch.setattr(api, "yf", _FakeYF())

    first = CLIENT.get("/api/v1/info?ticker=AAPL")
    assert first.status_code == 200
    body = first.json()
    assert body["ticker"] == "AAPL"
    assert body["sector"] == "Information Technology"
    assert body["currency"] == "USD"
    assert body["currency_symbol"] == "$"
    assert body["marketCapCurrency"] == "USD"
    assert body["marketCapCurrencySymbol"] == "$"
    assert body["marketCap"] == pytest.approx(3_150_000_000_000.0)
    assert body["fiftyTwoWeekHigh"] == pytest.approx(242.5)
    assert body["fiftyTwoWeekLow"] == pytest.approx(168.0)
    assert body["peRatio"] is None or (
        isinstance(body["peRatio"], float)
        and body["peRatio"] > 0
        and body["peSource"] == "offline_batch"
    )
    assert calls["ticker"] == 1

    # Second call served from positive TTL cache.
    second = CLIENT.get("/api/v1/info?ticker=AAPL")
    assert second.status_code == 200
    assert second.json() == body
    assert calls["ticker"] == 1


def test_info_london_gbp_mixed_and_whole_pound_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: dict[str, dict[str, Any]] = {
        # Mixed GBp/GBP: prev_close in GBp (2780p), 52W range arrived in GBP (29.5 / 24.1).
        "SHEL.L": {
            "market_cap": 165_000_000_000.0,
            "previous_close": 2780.0,
            "year_high": 29.50,
            "year_low": 24.10,
            "day_high": 2805.0,
            "day_low": 2760.0,
            "currency": "GBp",
        },
        # Whole-GBP feed: all share prices in GBP (£118.40), market cap in GBP.
        "AZN.L": {
            "market_cap": 185_000_000_000.0,
            "previous_close": 118.4,
            "year_high": 130.0,
            "year_low": 99.5,
            "day_high": 120.2,
            "day_low": 117.1,
            "currency": "GBP",
        },
    }

    class _FakeYF:
        def Ticker(self, symbol: str) -> Any:
            return SimpleNamespace(fast_info=payloads[symbol])

    monkeypatch.setattr(api, "yf", _FakeYF())

    shel = CLIENT.get("/api/v1/info?ticker=SHEL.L")
    assert shel.status_code == 200
    shel_body = shel.json()
    assert shel_body["currency"] == "GBp"
    assert shel_body["currency_symbol"] == "p"
    assert shel_body["marketCapCurrency"] == "GBP"
    assert shel_body["marketCapCurrencySymbol"] == "£"
    assert shel_body["marketCap"] == pytest.approx(165_000_000_000.0)
    assert shel_body["previousClose"] == pytest.approx(2780.0)
    assert shel_body["fiftyTwoWeekHigh"] == pytest.approx(2950.0)
    assert shel_body["fiftyTwoWeekLow"] == pytest.approx(2410.0)
    assert shel_body["sector"] == "Energy"

    azn = CLIENT.get("/api/v1/info?ticker=AZN.L")
    assert azn.status_code == 200
    azn_body = azn.json()
    assert azn_body["currency"] == "GBp"
    assert azn_body["marketCapCurrency"] == "GBP"
    assert azn_body["marketCap"] == pytest.approx(185_000_000_000.0)
    assert azn_body["previousClose"] == pytest.approx(11840.0)
    assert azn_body["fiftyTwoWeekHigh"] == pytest.approx(13000.0)
    assert azn_body["fiftyTwoWeekLow"] == pytest.approx(9950.0)
    assert azn_body["dayHigh"] == pytest.approx(12020.0)
    assert azn_body["dayLow"] == pytest.approx(11710.0)


def test_info_london_fast_info_pence_market_cap_converted_to_gbp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live yfinance fast_info returns shares * last_price in GBp (100x GBP); convert to GBP."""

    class _FakeYF:
        def Ticker(self, symbol: str) -> Any:
            return SimpleNamespace(
                fast_info={
                    "currency": "GBp",
                    "last_price": 3552.0,
                    "previous_close": 3540.0,
                    "shares": 5_645_167_459,
                    "market_cap": 20_051_634_814_368.0,
                    "year_high": 4110.0,
                    "year_low": 2553.77,
                }
            )

    monkeypatch.setattr(api, "yf", _FakeYF())

    resp = CLIENT.get("/api/v1/info?ticker=SHEL.L")
    assert resp.status_code == 200
    body = resp.json()
    assert body["currency"] == "GBp"
    assert body["currency_symbol"] == "p"
    assert body["marketCapCurrency"] == "GBP"
    assert body["marketCapCurrencySymbol"] == "£"
    assert body["marketCap"] == pytest.approx(200_516_348_143.68, rel=1e-6)
    assert body["fiftyTwoWeekHigh"] == pytest.approx(4110.0)
    assert body["fiftyTwoWeekLow"] == pytest.approx(2553.77)


def test_info_london_usd_overrides_cpg_and_ihg_without_pence_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeYF:
        def Ticker(self, symbol: str) -> Any:
            return SimpleNamespace(
                fast_info={
                    "market_cap": 52_000_000_000.0,
                    "previous_close": 34.50,
                    "year_high": 38.20,
                    "year_low": 26.80,
                    "currency": "USD",
                }
            )

    monkeypatch.setattr(api, "yf", _FakeYF())

    for sym in ("CPG.L", "IHG.L"):
        resp = CLIENT.get(f"/api/v1/info?ticker={sym}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["currency"] == "USD"
        assert body["currency_symbol"] == "$"
        assert body["marketCapCurrency"] == "USD"
        assert body["marketCapCurrencySymbol"] == "$"
        assert body["fiftyTwoWeekHigh"] == pytest.approx(38.20)
        assert body["fiftyTwoWeekLow"] == pytest.approx(26.80)
        assert body["previousClose"] == pytest.approx(34.50)


def test_info_negative_cache_prevents_repeated_upstream_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    class _FailingYF:
        def Ticker(self, symbol: str) -> Any:
            calls["count"] += 1
            raise RuntimeError("yfinance throttled")

    monkeypatch.setattr(api, "yf", _FailingYF())

    first = CLIENT.get("/api/v1/info?ticker=NVDA")
    second = CLIENT.get("/api/v1/info?ticker=NVDA")
    assert first.status_code == 503
    assert second.status_code == 503
    assert calls["count"] == 1


def test_info_reads_disk_universe_metadata_json_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_backend = tmp_path / "backend"
    (fake_backend / "routes").mkdir(parents=True)
    (fake_backend / "data").mkdir(parents=True)
    fake_json = fake_backend / "data" / "universe_metadata.json"
    fake_json.write_text(
        json.dumps(
            {
                "tickers": {
                    "MSFT": {
                        "ticker": "MSFT",
                        "name": "Microsoft Corporation (Catalog)",
                        "exchange": "NASDAQ",
                        "exchange_mic": "XNAS",
                        "sector": "Information Technology",
                        "industry": "Enterprise Software",
                        "cik": "0000789019",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(market_route, "__file__", str(fake_backend / "routes" / "market.py"))
    monkeypatch.setattr(market_route, "_catalog_mtime_ns", None)
    market_route._catalog_cache.clear()

    class _FakeYF:
        def Ticker(self, symbol: str) -> Any:
            return SimpleNamespace(
                fast_info={"market_cap": 3_000_000_000_000.0, "previous_close": 420.0}
            )

    monkeypatch.setattr(api, "yf", _FakeYF())

    resp = CLIENT.get("/api/v1/info?ticker=MSFT")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Microsoft Corporation (Catalog)"
    assert body["sector"] == "Information Technology"
    assert body["industry"] == "Enterprise Software"
    assert body["exchange"] == "NASDAQ"
    market_route._catalog_cache.clear()
    monkeypatch.setattr(market_route, "_catalog_mtime_ns", None)


# ---------------------------------------------------------------------------
# PR 6B: Bounded Intraday 24H Bars
# ---------------------------------------------------------------------------


def test_alpaca_fetch_intraday_bars_latest_ny_session_and_sparse_guard() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        if "SPARSE" in str(request.url):
            # Fewer than 5 valid bars -> must return []
            return httpx.Response(
                200,
                json={
                    "bars": [
                        {"t": "2026-11-20T14:30:00Z", "c": 100.0},
                        {"t": "2026-11-20T14:45:00Z", "c": 101.0},
                        {"t": "2026-11-20T15:00:00Z", "c": 102.0},
                        {"t": "2026-11-20T15:15:00Z", "c": 103.0},
                    ]
                },
            )
        # Multi-session payload with an invalid bar on the latest session.
        return httpx.Response(
            200,
            json={
                "bars": [
                    {"t": "2026-11-19T19:30:00Z", "c": 220.0},
                    {"t": "2026-11-19T19:45:00Z", "c": 221.0},
                    {"t": "2026-11-20T14:30:00Z", "c": 225.0},
                    {"t": "2026-11-20T14:45:00Z", "c": 225.5},
                    {"t": "2026-11-20T15:00:00Z", "c": -1.0},  # invalid close dropped
                    {"t": "2026-11-20T15:15:00Z", "c": 226.0},
                    {"t": "2026-11-20T15:30:00Z", "c": 226.4},
                    {"t": "2026-11-20T15:45:00Z", "c": 227.1},
                    {"t": "2026-11-20T16:00:00Z", "c": 227.8},
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    provider = AlpacaProvider(key_id="k", secret_key="s", client=client)
    try:
        bars = provider.fetch_intraday_bars("AAPL")
        assert len(bars) == 6
        assert all(b["t"].startswith("2026-11-20") for b in bars)
        assert bars[0]["c"] == pytest.approx(225.0)
        assert bars[-1]["c"] == pytest.approx(227.8)

        assert provider.fetch_intraday_bars("SPARSE") == []
    finally:
        client.close()


def test_intraday_route_caches_positive_and_negative_and_skips_lse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls: list[str] = []

    class _FakeProvider:
        configured = True

        def fetch_intraday_bars(self, symbol: str) -> list[dict[str, Any]]:
            provider_calls.append(symbol)
            if symbol == "MSFT":
                # Sparse (< 5 bars) -> triggers 60s negative cache
                return [
                    {"t": "2026-11-20T14:30:00Z", "c": 420.0},
                    {"t": "2026-11-20T14:45:00Z", "c": 421.0},
                ]
            return [
                {"t": f"2026-11-20T14:{minute:02d}:00Z", "c": 225.0 + idx * 0.25}
                for idx, minute in enumerate((0, 10, 20, 30, 40, 50))
            ]

    monkeypatch.setattr(market_route, "_alpaca_intraday_provider", lambda: _FakeProvider())

    # LSE returns unavailable immediately without hitting Alpaca.
    lse_resp = CLIENT.get("/api/v1/intraday?ticker=SHEL.L")
    assert lse_resp.status_code == 200
    assert lse_resp.json() == {
        "ticker": "SHEL.L",
        "session": None,
        "bars": [],
        "bar_count": 0,
        "available": False,
    }
    assert provider_calls == []

    # US positive path + 60s positive cache.
    aapl_1 = CLIENT.get("/api/v1/intraday?ticker=AAPL")
    aapl_2 = CLIENT.get("/api/v1/intraday?ticker=AAPL")
    assert aapl_1.status_code == 200
    assert aapl_1.headers.get("cache-control") == "public, max-age=60"
    body = aapl_1.json()
    assert body["available"] is True
    assert body["bar_count"] == 6
    assert body["session"] == "2026-11-20"
    assert aapl_2.json() == body
    assert provider_calls == ["AAPL"]

    # Sparse US response triggers 60s negative cache.
    msft_1 = CLIENT.get("/api/v1/intraday?ticker=MSFT")
    msft_2 = CLIENT.get("/api/v1/intraday?ticker=MSFT")
    assert msft_1.status_code == 200
    assert msft_1.json()["available"] is False
    assert msft_2.json()["available"] is False
    assert provider_calls == ["AAPL", "MSFT"]


def test_history_intraday_available_flag_does_not_block_on_cold_intraday(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2026-11-10", periods=10, freq="B")
    frame = pd.DataFrame({"Close": np.linspace(200.0, 210.0, len(idx))}, index=idx)

    def _fake_daily(symbol: str) -> MarketDataResult:
        del symbol
        return MarketDataResult(
            frame=frame,
            provider="alpaca",
            feed="iex",
            adjustment="split",
            data_as_of="2026-11-23",
            data_fingerprint="f" * 64,
            cache_status="hit",
        )

    class _NoCallProvider:
        configured = True

        def fetch_intraday_bars(self, symbol: str) -> list[dict[str, Any]]:
            raise AssertionError("GET /api/v1/history must not block on cold intraday fetch")

    monkeypatch.setattr(market_route, "_fetch_history_daily", _fake_daily)
    monkeypatch.setattr(market_route, "_alpaca_intraday_provider", lambda: _NoCallProvider())

    us_resp = CLIENT.get("/api/v1/history?ticker=AAPL")
    assert us_resp.status_code == 200
    us_body = us_resp.json()
    assert us_body["intraday"] is None
    assert us_body["intraday_available"] is True

    lse_resp = CLIENT.get("/api/v1/history?ticker=SHEL.L")
    assert lse_resp.status_code == 200
    assert lse_resp.json()["intraday_available"] is False


# ---------------------------------------------------------------------------
# PR 6C: Bounded News Pipeline & Rate-Limited SEC EDGAR
# ---------------------------------------------------------------------------


def test_news_pooled_client_cache_bounds_and_shutdown_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert market_news._cache.maxsize == 256
    client_a = market_news._get_http_client()
    client_b = market_news._get_http_client()
    assert client_a is client_b
    assert not client_a.is_closed

    market_news.close_news_client()
    assert client_a.is_closed
    assert market_news._http_client is None

    # Shutdown hook closes the pooled news client cleanly.
    client_c = market_news._get_http_client()
    assert not client_c.is_closed
    api._shutdown_resources()
    assert client_c.is_closed


def test_sec_rate_limiter_enforces_strictly_under_10_req_per_sec_concurrently() -> None:
    limiter_inst = market_news._SecRateLimiter(min_interval_seconds=0.12)
    timestamps: list[float] = []
    ts_lock = threading.Lock()

    def _worker() -> None:
        assert limiter_inst.acquire(max_wait_seconds=2.0) is True
        with ts_lock:
            timestamps.append(time.monotonic())

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    timestamps.sort()
    assert len(timestamps) == 4
    deltas = [b - a for a, b in zip(timestamps[:-1], timestamps[1:], strict=True)]
    assert all(delta >= 0.105 for delta in deltas), f"Intervals violated <10 req/s: {deltas}"

    # Tight budget rejects immediately without waiting past budget.
    assert limiter_inst.acquire(max_wait_seconds=0.005) is False


def test_sec_edgar_cik_cache_wbd_override_and_compliant_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert market_news.resolve_ticker_cik("WBD") == "0001437107"
    assert market_news.resolve_ticker_cik("GOOGL") == "0001652044"
    assert market_news.resolve_ticker_cik("GOOG") == "0001652044"
    assert market_news.resolve_ticker_cik("XOM") == "0002115436"
    assert market_news.resolve_ticker_cik("SHEL.L") is None

    seen_requests: list[tuple[str, str]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append((str(request.url), request.headers.get("User-Agent", "")))
        return httpx.Response(
            200,
            json={
                "filings": {
                    "recent": {
                        "form": ["8-K", "4", "10-Q"],
                        "filingDate": ["2026-11-18", "2026-11-17", "2026-10-30"],
                        "acceptanceDateTime": [
                            "2026-11-18T21:15:00.000Z",
                            "2026-11-17T18:00:00.000Z",
                            "2026-10-30T20:10:00.000Z",
                        ],
                        "accessionNumber": [
                            "0001652044-26-000101",
                            "0001652044-26-000099",
                            "0001652044-26-000080",
                        ],
                        "primaryDocument": ["goog-8k.htm", "form4.xml", "goog-10q.htm"],
                        "primaryDocDescription": ["FORM 8-K", "FORM 4", "FORM 10-Q"],
                        "items": ["2.02,8.01", "", ""],
                    }
                }
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(_handler))
    monkeypatch.setattr(market_news, "_get_http_client", lambda: mock_client)
    try:
        googl_filings = market_news._fetch_from_sec_edgar("GOOGL", timeout_seconds=2.0)
        goog_filings = market_news._fetch_from_sec_edgar("GOOG", timeout_seconds=2.0)
    finally:
        mock_client.close()

    # Share classes GOOGL and GOOG share one CIK-keyed SEC call.
    assert len(seen_requests) == 1
    req_url, ua = seen_requests[0]
    assert req_url.endswith("/submissions/CIK0001652044.json")
    assert "SignalSeven" in ua and "@" in ua

    assert len(googl_filings) == 2  # Form 4 excluded; 8-K and 10-Q included
    assert googl_filings[0]["kind"] == "sec_filing"
    assert googl_filings[0]["form"] == "8-K"
    assert googl_filings[0]["filing_date"] == "2026-11-18"
    assert googl_filings[0]["cik"] == "0001652044"
    assert len(goog_filings) == 2
    assert goog_filings[0]["form"] == "8-K"


def test_news_route_distinct_sec_filings_negative_cache_and_shared_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_article = {
        "id": "art-1",
        "title": "Apple unveils new chip lineup",
        "summary": "Apple announced next-generation silicon.",
        "source": "Market Wire",
        "published_at": "2026-11-20T18:00:00Z",
        "url": "https://example.com/aapl-chip",
        "provider": "yahoo",
        "kind": "news",
        "sentiment": {
            "compound": 0.4,
            "positive": 0.4,
            "negative": 0.0,
            "neutral": 0.6,
            "label": "bullish",
        },
    }
    sample_filing = {
        "id": "sec-1",
        "title": "AAPL Form 8-K: Results of Operations",
        "summary": "Official U.S. SEC 8-K filing for AAPL.",
        "source": "SEC EDGAR",
        "published_at": "2026-11-19T21:30:00Z",
        "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019326000001/aapl-8k.htm",
        "provider": "sec_edgar",
        "kind": "sec_filing",
        "form": "8-K",
        "filing_date": "2026-11-19",
        "cik": "0000320193",
        "sentiment": {
            "compound": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "label": "neutral",
        },
    }
    monkeypatch.setattr(market_news, "_fetch_from_yahoo", lambda sym: [dict(sample_article)])
    monkeypatch.setattr(
        market_news, "_fetch_from_sec_edgar", lambda sym, timeout_seconds=4.0: [dict(sample_filing)]
    )
    monkeypatch.setattr(market_news, "get_exchange_market_close", lambda sym: None)

    resp = CLIENT.get("/api/v1/news?ticker=AAPL")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "available"
    assert len(body["items"]) == 1
    assert body["articles"] == body["items"]
    assert len(body["sec_filings"]) == 1
    assert body["sec_filings"][0]["kind"] == "sec_filing"
    assert body["sec_filings"][0]["form"] == "8-K"

    # Negative cache prevents repeated empty lookups for 60s.
    market_news.clear_news_cache()
    yahoo_calls = {"count": 0}

    def _empty_yahoo(sym: str) -> list[dict[str, Any]]:
        yahoo_calls["count"] += 1
        return []

    monkeypatch.setattr(market_news, "_fetch_from_yahoo", _empty_yahoo)
    monkeypatch.setattr(market_news, "_fetch_from_sec_edgar", lambda sym, timeout_seconds=4.0: [])
    monkeypatch.setattr(market_news, "_fetch_from_cache", lambda sym: [])

    res1 = market_news.fetch_recent_news("ZZZZ", include_sec_filings=True)
    res2 = market_news.fetch_recent_news("ZZZZ", include_sec_filings=True)
    assert res1["status"] == "unavailable"
    assert res2["status"] == "unavailable"
    assert yahoo_calls["count"] == 1

    # Shared deadline skips slow downstream fallbacks once budget is exhausted.
    market_news.clear_news_cache()
    downstream_called = {"yahoo": False, "sec": False}

    def _slow_alpaca(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        time.sleep(0.16)
        raise RuntimeError("alpaca timeout")

    def _spy_yahoo(sym: str) -> list[dict[str, Any]]:
        downstream_called["yahoo"] = True
        return []

    def _spy_sec(sym: str, timeout_seconds: float = 4.0) -> list[dict[str, Any]]:
        downstream_called["sec"] = True
        return []

    monkeypatch.setattr(market_news, "_fetch_from_alpaca", _slow_alpaca)
    monkeypatch.setattr(market_news, "_fetch_from_yahoo", _spy_yahoo)
    monkeypatch.setattr(market_news, "_fetch_from_sec_edgar", _spy_sec)

    t0 = time.monotonic()
    bounded = market_news.fetch_recent_news(
        "SLOW",
        key_id="k",
        secret_key="s",
        timeout_seconds=0.12,
        include_sec_filings=True,
    )
    elapsed = time.monotonic() - t0
    assert bounded["status"] == "unavailable"
    assert downstream_called == {"yahoo": False, "sec": False}
    assert elapsed < 1.0


# ---------------------------------------------------------------------------
# PR 6D: Versioned V2 Cohort, Collector Batching & Indexed Ledger Summary
# ---------------------------------------------------------------------------


def test_live_universe_v1_preserved_and_v2_cohort_contracts() -> None:
    assert len(LIVE_UNIVERSE_V1) == 15
    assert LIVE_EXPECTED_RECORD_COUNT == 60
    assert len(LIVE_COLLECTION_ITEMS_V1) == 60
    assert "PLD" in LIVE_UNIVERSE_V1 and "NMM" in LIVE_UNIVERSE_V1

    assert LIVE_UNIVERSE_VERSION_V2 == "live_v2_2026_11"
    assert len(LIVE_UNIVERSE_V2_US) == 20
    assert len(LIVE_UNIVERSE_V2_LSE) == 10
    assert len(LIVE_UNIVERSE_V2) == 30
    assert len(set(LIVE_UNIVERSE_V2)) == 30
    assert len(LIVE_COLLECTION_ITEMS_V2) == 120
    assert "PLD" not in LIVE_UNIVERSE_V2 and "NMM" not in LIVE_UNIVERSE_V2
    assert all(not t.endswith(".L") for t in LIVE_UNIVERSE_V2_US)
    assert all(t.endswith(".L") for t in LIVE_UNIVERSE_V2_LSE)

    assert validate_live_collection_item("shel.l", 5, cohort="v2") == ("SHEL.L", 5)
    with pytest.raises(ValueError, match="frozen live universe"):
        validate_live_collection_item("SHEL.L", 5, cohort="v1")


def test_resolve_live_cohort_items_exchange_filter_and_resumable_batching() -> None:
    us_items = resolve_live_cohort_items("v2", exchange="NYSE")
    lse_items = resolve_live_cohort_items("v2", exchange="LSE")
    assert len(us_items) == 80
    assert len(lse_items) == 40
    assert all(not t.endswith(".L") for t, _ in us_items)
    assert all(t.endswith(".L") for t, _ in lse_items)

    # Partition all 120 V2 items into 5 disjoint resumable batches of 24.
    reconstructed: list[tuple[str, int]] = []
    for batch_idx in range(5):
        chunk = resolve_live_cohort_items(
            "v2", exchange="ALL", batch_size=24, batch_index=batch_idx
        )
        assert len(chunk) == 24
        reconstructed.extend(chunk)
    assert tuple(reconstructed) == LIVE_COLLECTION_ITEMS_V2
    assert resolve_live_cohort_items("v2", exchange="ALL", batch_size=24, batch_index=5) == ()


def test_collector_session_cutoffs_lse_vs_nyse_and_v2_batch_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collector_script.time, "sleep", lambda _s: None)
    # LSE scheduled window: 17:30 Europe/London on a trading Friday (2026-11-20 GMT == UTC).
    lse_ok, lse_reason = collection_session(
        datetime(2026, 11, 20, 17, 30, tzinfo=UTC), scheduled=True, exchange="LSE"
    )
    assert lse_ok == "2026-11-20"
    assert lse_reason is None

    lse_early, lse_early_reason = collection_session(
        datetime(2026, 11, 20, 16, 45, tzinfo=UTC), scheduled=True, exchange="LSE"
    )
    assert lse_early is None
    assert lse_early_reason == "outside_17_30_europe_london_lse_window"

    lse_wknd, lse_wknd_reason = collection_session(
        datetime(2026, 11, 22, 17, 30, tzinfo=UTC), scheduled=True, exchange="LSE"
    )
    assert lse_wknd is None
    assert lse_wknd_reason == "no_completed_lse_session_on_london_date"

    # CLI flags parse cleanly for V2 LSE batching.
    parsed = parse_args(
        [
            "--base-url",
            "https://api.example",
            "--mode",
            "live",
            "--cohort",
            "v2",
            "--exchange",
            "LSE",
            "--batch-size",
            "8",
            "--batch-index",
            "0",
        ]
    )
    assert parsed.cohort == "v2"
    assert parsed.exchange == "LSE"
    assert parsed.batch_size == 8
    assert parsed.batch_index == 0

    class _FakeBatchClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, Any]]] = []

        def request(
            self,
            method: str,
            path: str,
            *,
            params: dict[str, Any] | None = None,
            token: str | None = None,
            attempts: int = 2,
        ) -> ApiResult:
            del token, attempts
            params = dict(params or {})
            self.calls.append((method, path, params))
            if path == "/ready":
                return ApiResult(
                    200,
                    {
                        "status": "ready",
                        "dependencies": {
                            "market_data": {"status": "available"},
                            "forecast_ledger": {
                                "status": "available",
                                "backend": "postgresql",
                                "durable": True,
                                "required": True,
                            },
                        },
                    },
                )
            if path == "/api/v1/volatility/score-ledger":
                return ApiResult(200, {"scored_count": 1})
            ticker = str(params["ticker"])
            horizon = int(params["horizon"])
            fp = hashlib.sha256(f"{ticker}:{horizon}:2026-11-20".encode()).hexdigest()
            return ApiResult(
                200,
                {
                    "ticker": ticker,
                    "horizon": horizon,
                    "forecast": {"model": LIVE_MODEL_POLICY_V1[horizon]},
                    "evidence": {
                        "data_as_of": "2026-11-20",
                        "code_commit": "abc12345",
                        "forecast_fingerprint": fp,
                        "ledger_write": (
                            "recorded_live" if method == "POST" else "disabled_public_preview"
                        ),
                    },
                },
            )

    fake_client = _FakeBatchClient()
    preflight = run_preflight(
        fake_client,  # type: ignore[arg-type]
        expected_session="2026-11-20",
        run_timestamp="2026-11-20T17:30:00+00:00",
        interval_seconds=0,
        cohort="v2",
        exchange="LSE",
        batch_size=8,
        batch_index=0,
    )
    assert preflight["batch_status"] == "dry_run_passed"
    assert preflight["universe_version"] == LIVE_UNIVERSE_VERSION_V2
    assert preflight["exchange"] == "LSE"
    assert preflight["expected_record_count"] == 8
    assert preflight["succeeded_count"] == 8
    assert all(item["ticker"].endswith(".L") for item in preflight["items"])

    live = run_live_collection(
        fake_client,  # type: ignore[arg-type]
        preflight,
        token="secret",
        interval_seconds=0,
    )
    assert live["batch_status"] == "complete"
    assert live["succeeded_count"] == 8
    post_calls = [c for c in fake_client.calls if c[0] == "POST"]
    assert len(post_calls) == 8
    assert all(params.get("cohort") == LIVE_UNIVERSE_VERSION_V2 for _, _, params in post_calls)

    # Resumable settlement batch on LSE V2 slice.
    settle = run_settlement(
        fake_client,  # type: ignore[arg-type]
        token="secret",
        cohort="v2",
        exchange="LSE",
        batch_size=3,
        batch_index=0,
    )
    assert settle["status"] == "complete"
    assert settle["universe_version"] == LIVE_UNIVERSE_VERSION_V2
    assert [it["ticker"] for it in settle["items"]] == list(LIVE_UNIVERSE_V2_LSE[:3])


def test_collect_route_accepts_v2_cohort_and_rejects_lse_on_v1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = ForecastLedger(tmp_path / "collect_v2.db")
    monkeypatch.setenv("FORECAST_COLLECTOR_TOKEN", "test-collector-secret")
    monkeypatch.setattr(volatility_route, "get_forecast_ledger", lambda: ledger)

    def _fake_snapshot(symbol: str, *, allow_post_close_grace: bool = False) -> SimpleNamespace:
        del allow_post_close_grace
        return SimpleNamespace(
            ticker=symbol,
            snapshot_id="b" * 64,
            origin_date="2026-11-20",
            data_as_of="2026-11-20",
            origin_close=2780.0,
            data_provider="yfinance",
            data_feed="yfinance",
            price_adjustment="split",
            data_fingerprint="c" * 64,
            market_data_cache="miss",
            feature_names=("Return_1D", "Vol_C2C_20"),
            features=np.ones((60, 2), dtype=np.float32),
            causal_har_variance=np.full(len(VOLATILITY_HORIZONS), 0.04, dtype=np.float32),
            baseline_candidates={
                "rolling_c2c_5": np.full(len(VOLATILITY_HORIZONS), 0.04, dtype=np.float64),
                "rolling_c2c_20": np.full(len(VOLATILITY_HORIZONS), 0.04, dtype=np.float64),
                "rolling_c2c_60": np.full(len(VOLATILITY_HORIZONS), 0.03, dtype=np.float64),
                "riskmetrics_ewma_c2c": np.full(len(VOLATILITY_HORIZONS), 0.03, dtype=np.float64),
                "causal_log_har": np.full(len(VOLATILITY_HORIZONS), 0.025, dtype=np.float64),
                "garch_11": np.full(len(VOLATILITY_HORIZONS), 0.025, dtype=np.float64),
            },
            historical_dates=("2026-11-19", "2026-11-20"),
            historical_prices=np.array([2760.0, 2780.0]),
            future_dates=tuple(f"2026-12-{day:02d}" for day in range(1, 21)),
        )

    monkeypatch.setattr(volatility_route, "build_volatility_inference_snapshot", _fake_snapshot)
    headers = {"Authorization": "Bearer test-collector-secret"}

    # Default V1 rejects SHEL.L cleanly.
    v1_resp = CLIENT.post(
        "/api/v1/volatility/collect",
        params={"ticker": "SHEL.L", "horizon": 5},
        headers=headers,
    )
    assert v1_resp.status_code == 400

    # V2 cohort accepts SHEL.L and stamps live_universe_version == live_v2_2026_11.
    v2_resp = CLIENT.post(
        "/api/v1/volatility/collect",
        params={"ticker": "SHEL.L", "horizon": 5, "cohort": "v2"},
        headers=headers,
    )
    assert v2_resp.status_code == 200
    v2_body = v2_resp.json()
    assert v2_body["evidence"]["ledger_write"] == "recorded_live"
    assert v2_body["evidence"]["live_universe_version"] == LIVE_UNIVERSE_VERSION_V2
    assert len(ledger.get_ledger_entries(ticker="SHEL.L", record_source="live")) == 1


def test_ledger_summary_sql_grouping_cohort_filters_and_composite_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "summary_ledger.db"
    ledger = ForecastLedger(db_path)
    monkeypatch.setattr(volatility_route, "get_forecast_ledger", lambda: ledger)

    # Confirm composite summary index exists in SQLite.
    with sqlite3.connect(db_path) as conn:
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='forecast_ledger'"
            ).fetchall()
        }
    assert "idx_ledger_summary_group" in indexes

    # Seed scored + pending records across V1, V2, live, replay, and settlement versions.
    ledger.record_forecast(
        forecast_date="2026-09-02",
        ticker="MSFT",
        horizon=5,
        target_date="2026-09-09",
        model_name="rolling_mean",
        predicted_volatility=0.25,
        recent_realized_volatility=0.20,
        origin_price=420.0,
        lower_scenario_price=395.0,
        upper_scenario_price=445.0,
        record_source="live",
        model_version="deployable_v5:annualized_sigma_v1",
        data_as_of="2026-09-02",
    )
    ledger.record_forecast(
        forecast_date="2026-09-03",
        ticker="PLD",
        horizon=5,
        target_date="2026-09-10",
        model_name="rolling_mean",
        predicted_volatility=0.22,
        recent_realized_volatility=0.21,
        origin_price=120.0,
        lower_scenario_price=112.0,
        upper_scenario_price=128.0,
        record_source="live",
        model_version="deployable_v5:annualized_sigma_v1",
        data_as_of="2026-09-03",
    )
    ledger.record_forecast(
        forecast_date="2026-11-20",
        ticker="SHEL.L",
        horizon=5,
        target_date="2026-11-27",
        model_name="rolling_mean",
        predicted_volatility=0.24,
        recent_realized_volatility=0.20,
        origin_price=2780.0,
        lower_scenario_price=2620.0,
        upper_scenario_price=2940.0,
        record_source="live",
        model_version="deployable_v5:annualized_sigma_v1",
        data_as_of="2026-11-20",
    )
    ledger.record_forecast(
        forecast_date="2026-08-10",
        ticker="MSFT",
        horizon=5,
        target_date="2026-08-17",
        model_name="rolling_mean",
        predicted_volatility=0.21,
        recent_realized_volatility=0.20,
        origin_price=410.0,
        lower_scenario_price=388.0,
        upper_scenario_price=432.0,
        record_source="historical_replay",
        model_version="deployable_v5",
        data_as_of="2026-08-10",
    )

    # Mark MSFT live, SHEL.L live, and MSFT replay as scored directly in SQLite to verify exact SQL math.
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE forecast_ledger
            SET status='scored', actual_realized_volatility=0.28,
                forecast_error=-0.03, abs_error=0.03, qlike_loss=0.02
            WHERE ticker='MSFT' AND record_source='live'
            """
        )
        conn.execute(
            """
            UPDATE forecast_ledger
            SET status='scored', actual_realized_volatility=0.18,
                forecast_error=0.06, abs_error=0.06, qlike_loss=0.08
            WHERE ticker='SHEL.L' AND record_source='live'
            """
        )
        conn.execute(
            """
            UPDATE forecast_ledger
            SET status='scored', actual_realized_volatility=0.23,
                forecast_error=-0.02, abs_error=0.02, qlike_loss=0.01
            WHERE ticker='MSFT' AND record_source='historical_replay'
            """
        )
        conn.commit()

    # 1. Unfiltered summary partitions cleanly without double-counting.
    all_resp = CLIENT.get("/api/v1/volatility/ledger/summary")
    assert all_resp.status_code == 200
    all_body = all_resp.json()
    assert all_body["total_groups"] == 3
    by_key = {
        (
            g["cohort"],
            g["record_source"],
            g["horizon"],
            g["policy_version"],
            g["settlement_version"],
        ): g
        for g in all_body["groups"]
    }
    v1_live = by_key[(LIVE_UNIVERSE_VERSION, "live", 5, "deployable_v5", "annualized_sigma_v1")]
    assert v1_live["total_forecasts"] == 2
    assert v1_live["scored_forecasts"] == 1
    assert v1_live["pending_forecasts"] == 1
    assert v1_live["distinct_tickers"] == 2
    assert v1_live["mean_mae"] == pytest.approx(0.03)
    assert v1_live["mean_rmse"] == pytest.approx(0.03)
    assert v1_live["mean_qlike"] == pytest.approx(0.02)
    assert v1_live["direction_accuracy_pct"] == pytest.approx(100.0)

    v2_live = by_key[(LIVE_UNIVERSE_VERSION_V2, "live", 5, "deployable_v5", "annualized_sigma_v1")]
    assert v2_live["total_forecasts"] == 1
    assert v2_live["scored_forecasts"] == 1
    assert v2_live["distinct_tickers"] == 1
    assert v2_live["direction_accuracy_pct"] == pytest.approx(0.0)

    v1_replay = by_key[
        (LIVE_UNIVERSE_VERSION, "historical_replay", 5, "deployable_v5", "legacy_unversioned")
    ]
    assert v1_replay["scored_forecasts"] == 1
    assert v1_replay["direction_eligible_forecasts"] == 0
    assert v1_replay["direction_accuracy_pct"] is None

    # 2. Cohort=v1 filter includes MSFT + PLD, excludes SHEL.L.
    v1_resp = CLIENT.get("/api/v1/volatility/ledger/summary?cohort=v1&record_source=live")
    assert v1_resp.status_code == 200
    v1_groups = v1_resp.json()["groups"]
    assert len(v1_groups) == 1
    assert v1_groups[0]["cohort"] == LIVE_UNIVERSE_VERSION
    assert v1_groups[0]["total_forecasts"] == 2
    assert v1_groups[0]["distinct_tickers"] == 2

    # 3. Cohort=v2 filter includes overlapping MSFT + SHEL.L, excludes V1-only PLD.
    v2_resp = CLIENT.get("/api/v1/volatility/ledger/summary?cohort=v2&record_source=live")
    assert v2_resp.status_code == 200
    v2_groups = v2_resp.json()["groups"]
    assert len(v2_groups) == 1
    assert v2_groups[0]["cohort"] == LIVE_UNIVERSE_VERSION_V2
    assert v2_groups[0]["total_forecasts"] == 2
    assert v2_groups[0]["scored_forecasts"] == 2
    assert v2_groups[0]["distinct_tickers"] == 2
    assert v2_groups[0]["mean_mae"] == pytest.approx(0.045)
    assert v2_groups[0]["mean_rmse"] == pytest.approx(math.sqrt((0.03**2 + 0.06**2) / 2.0))
    assert v2_groups[0]["direction_accuracy_pct"] == pytest.approx(50.0)
