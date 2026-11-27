"""TDD tests for non-blocking aggregated news sentiment context on /api/v1/forecast."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app
from routes import simple_forecast as forecast_route
from routes.common import limiter
from services import market_news

CLIENT = TestClient(app)


def _synthetic_ohlcv(rows: int = 660) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-02", periods=rows)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, size=rows)))
    open_ = close * np.exp(rng.normal(0.0, 0.003, size=rows))
    spread = np.abs(rng.normal(0.006, 0.002, size=rows))
    df = pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * (1.0 + spread),
            "Low": np.minimum(open_, close) * (1.0 - spread),
            "Close": close,
            "Volume": rng.integers(500_000, 5_000_000, size=rows).astype(float),
        },
        index=idx,
    )
    df.attrs.update(
        {
            "data_provider": "test",
            "data_feed": "test",
            "price_adjustment": "split",
            "data_as_of": idx[-1].date().isoformat(),
        }
    )
    return df


@pytest.fixture(autouse=True)
def _reset_news_and_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(limiter, "enabled", False)
    market_news.clear_news_cache()
    yield
    market_news.clear_news_cache()


def test_summarize_cached_news_context_is_non_blocking_on_cold_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("summarize_cached_news_context must never call network fetchers")

    monkeypatch.setattr(market_news, "_fetch_from_alpaca", _boom)
    monkeypatch.setattr(market_news, "_fetch_from_yahoo", _boom)
    monkeypatch.setattr(market_news, "_fetch_from_sec_edgar", _boom)

    started = time.perf_counter()
    summary = market_news.summarize_cached_news_context("MSFT")
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert elapsed_ms < 25.0
    assert summary["status"] == "not_cached"
    assert summary["sentiment_label"] == "unavailable"
    assert summary["mean_compound"] is None
    assert summary["headline_count"] == 0
    assert summary["sec_filing_count"] == 0
    assert summary["used_in_point_model"] is False


@pytest.mark.parametrize(
    ("compounds", "expected_label", "counts"),
    [
        ([0.42, 0.28, 0.15], "bullish", (2, 0, 1)),
        ([-0.38, -0.22, -0.15], "bearish", (0, 2, 1)),
        ([0.1501, -0.1501, 0.0], "neutral", (1, 1, 1)),
        ([0.08] * 10, "bullish", (0, 0, 10)),
        ([-0.08] * 10, "bearish", (0, 0, 10)),
        ([0.0799, 0.08, 0.08], "neutral", (0, 0, 3)),
        ([-0.0799, -0.08, -0.08], "neutral", (0, 0, 3)),
    ],
)
def test_summarize_cached_news_context_aggregates_sentiment_and_sec_filings(
    compounds: list[float],
    expected_label: str,
    counts: tuple[int, int, int],
) -> None:
    items = [
        {
            "id": f"n-{i}",
            "title": f"Headline {i}",
            "published_at": f"2026-11-20T16:00:{i:02d}Z",
            "sentiment": c,
            "provider": "yahoo",
        }
        for i, c in enumerate(compounds)
    ]
    sec_filings = [
        {
            "id": "sec-1",
            "title": "MSFT Form 8-K",
            "form": "8-K",
            "published_at": "2026-11-24T21:00:00Z",
            "kind": "sec_filing",
        }
    ]
    with market_news._lock:
        market_news._cache["MSFT"] = {
            "status": "available",
            "items": items,
            "articles": items,
            "sec_filings": sec_filings,
            "provider": "yahoo",
            "as_of": "2026-11-25T18:00:00Z",
        }

    summary = market_news.summarize_cached_news_context("msft")
    assert summary["status"] == "available"
    assert summary["sentiment_label"] == expected_label
    assert summary["mean_compound"] == round(sum(compounds) / len(compounds), 4)
    assert (summary["bullish_count"], summary["bearish_count"], summary["neutral_count"]) == counts
    assert summary["headline_count"] == len(compounds)
    assert summary["sec_filing_count"] == 1
    assert summary["provider"] == "yahoo"
    assert summary["latest_published_at"] == "2026-11-24T21:00:00Z"
    assert summary["used_in_point_model"] is False


def test_sec_fallback_counts_each_filing_once_and_has_no_headline_sentiment(monkeypatch) -> None:
    published_at = "2026-11-24T21:05:00Z"
    recent = {
        "form": ["8-K"],
        "acceptanceDateTime": [published_at],
        "accessionNumber": ["0000789019-26-000101"],
        "primaryDocument": ["msft-8k.htm"],
        "primaryDocDescription": ["Record revenue growth"],
    }
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"filings": {"recent": recent}})
    )
    monkeypatch.setattr(market_news, "_fetch_from_yahoo", lambda _: [])
    monkeypatch.setattr(market_news, "get_exchange_market_close", lambda _: None)
    with httpx.Client(transport=transport) as client:
        monkeypatch.setattr(market_news, "_get_http_client", lambda: client)
        feed = market_news.fetch_recent_news("MSFT", include_sec_filings=True)

    assert feed["items"] == feed["sec_filings"]
    assert feed["items"][0]["sentiment_badge"] == "Bullish"
    assert market_news.summarize_cached_news_context("MSFT") == {
        "status": "available",
        "sentiment_label": "unavailable",
        "mean_compound": None,
        "headline_count": 0,
        "sec_filing_count": 1,
        "bullish_count": 0,
        "bearish_count": 0,
        "neutral_count": 0,
        "latest_published_at": published_at,
        "provider": "sec_edgar",
        "used_in_point_model": False,
    }


@pytest.mark.parametrize(
    "identity",
    [{"id": "sec-1"}, {"url": "https://www.sec.gov/filing.htm"}, {"title": "MSFT Form 8-K"}],
)
def test_mixed_cache_counts_filing_identities_across_both_collections(identity: dict) -> None:
    with market_news._lock:
        market_news._cache["MSFT"] = {
            "items": [
                {"id": "news", "sentiment": -0.4},
                {**identity, "sentiment": 0.8},
                {"id": "sec-2", "provider": "sec_edgar", "sentiment": 0.7},
            ],
            "sec_filings": [
                {**identity, "form": "8-K"},
                {"id": "sec-3", "form": "10-Q"},
            ],
            "provider": "alpaca",
        }

    summary = market_news.summarize_cached_news_context("MSFT")
    assert summary["headline_count"] == 1
    assert summary["sec_filing_count"] == 3
    assert summary["mean_compound"] == -0.4
    assert summary["sentiment_label"] == "bearish"
    assert summary["bearish_count"] == 1
    assert summary["bullish_count"] == summary["neutral_count"] == 0


def test_distinct_documents_drive_both_news_and_forecast(monkeypatch) -> None:
    shared_url = "https://wire.example/earnings"
    filing_url = "https://www.sec.gov/Archives/edgar/data/report.htm"
    items = [
        {"id": "wire-old", "url": shared_url, "published_at": "2026-11-20", "sentiment": 0.8},
        {"id": "wire-new", "url": shared_url, "published_at": "2026-11-22", "sentiment": -0.2},
        {"id": "syndicated", "url": filing_url, "sentiment": 0.8},
    ]
    filings = [{"id": "sec-1", "url": filing_url, "form": "8-K"}]
    monkeypatch.setattr(market_news, "_fetch_from_yahoo", lambda _: items)
    monkeypatch.setattr(market_news, "_call_sec_edgar", lambda *_: filings)
    monkeypatch.setattr(market_news, "get_exchange_market_close", lambda _: None)
    feed = market_news.fetch_recent_news("MSFT", include_sec_filings=True)
    assert [item["id"] for item in feed["items"]] == ["wire-new"]
    assert feed["sec_filings"] == filings
    digest = market_news.summarize_cached_news_context("MSFT")
    assert (digest["headline_count"], digest["sec_filing_count"]) == (1, 1)
    assert digest["sentiment_label"] == "bearish"
    assert digest["mean_compound"] == -0.2


def test_missing_primary_document_links_to_each_accession(monkeypatch) -> None:
    accessions = ["0000789019-26-000101", "0000789019-26-000102"]
    recent = {"form": ["8-K", "10-Q"], "accessionNumber": accessions, "primaryDocument": [None]}
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"filings": {"recent": recent}})
    )
    with httpx.Client(transport=transport) as client:
        monkeypatch.setattr(market_news, "_get_http_client", lambda: client)
        filings = market_news._fetch_from_sec_edgar("MSFT")
    assert {item["url"] for item in filings} == {
        f"https://www.sec.gov/Archives/edgar/data/789019/{acc.replace('-', '')}/{acc}-index.htm"
        for acc in accessions
    }


@pytest.mark.parametrize("with_wire_copy", [False, True])
def test_edgar_accessions_are_distinct_before_the_five_filing_cap(monkeypatch, with_wire_copy):
    sequence = [1, 1, 2, 3, 4, 5, 6]
    recent = {
        "form": ["8-K"] * len(sequence),
        "accessionNumber": [f"0000789019-26-{n:06d}" for n in sequence],
        "filingDate": [f"2026-11-{20 - n:02d}" for n in sequence],
    }
    wire = {
        "id": "yahoo-primary-doc",
        "url": "https://www.sec.gov/Archives/edgar/data/0000789019/000078901926000001/msft-8k.htm",
        "sentiment": 0.9,
    }
    monkeypatch.setattr(
        market_news, "_fetch_from_yahoo", lambda _: [wire] if with_wire_copy else []
    )
    monkeypatch.setattr(market_news, "get_exchange_market_close", lambda _: None)
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"filings": {"recent": recent}})
    )
    with httpx.Client(transport=transport) as client:
        monkeypatch.setattr(market_news, "_get_http_client", lambda: client)
        feed = market_news.fetch_recent_news("MSFT", include_sec_filings=True)
    assert [it["id"] for it in feed["sec_filings"]] == [
        f"sec-0000789019-26-{n:06d}" for n in range(1, 6)
    ]
    assert feed["items"] == ([] if with_wire_copy else feed["sec_filings"])
    digest = market_news.summarize_cached_news_context("MSFT")
    assert digest["headline_count"] == 0
    assert digest["sec_filing_count"] == 5
    assert digest["sentiment_label"] == "unavailable"


@pytest.mark.parametrize("with_score", [False, True])
def test_invalid_scores_are_excluded_from_sentiment_only(with_score: bool) -> None:
    items = [
        {"title": "No score"},
        {"sentiment": None},
        {"sentiment": True},
        {"sentiment": {"compound": False}},
        {"sentiment": {}},
        {"sentiment": "unknown"},
        {"sentiment": float("nan")},
        {"sentiment": float("inf")},
        {"sentiment": -1.1},
        {"sentiment": 10**400},
    ]
    if with_score:
        items.append({"sentiment": {"compound": "0.25"}})
    with market_news._lock:
        market_news._cache["MSFT"] = {"items": items, "provider": "alpaca"}

    summary = market_news.summarize_cached_news_context("MSFT")
    assert summary["status"] == "available"
    assert summary["headline_count"] == len(items)
    assert summary["mean_compound"] == (0.25 if with_score else None)
    assert summary["sentiment_label"] == ("bullish" if with_score else "unavailable")
    assert summary["bullish_count"] == int(with_score)
    assert summary["neutral_count"] == summary["bearish_count"] == 0


@pytest.mark.parametrize(
    ("earlier", "later"),
    [
        ("2026-11-25T00:30:00+02:00", "2026-11-24T18:45:00-05:00"),
        ("2026-11-24T21:00:00Z", "2026-11-24T21:00:00.125Z"),
    ],
)
def test_news_feed_and_cached_summary_use_publication_instants(monkeypatch, earlier, later) -> None:
    # The feed limit must retain the newest headline even when its string sorts last.
    items = [{"id": str(i), "published_at": earlier} for i in range(10)]
    items.append({"id": "newest", "published_at": later})
    monkeypatch.setattr(market_news, "_fetch_from_yahoo", lambda _: items)
    monkeypatch.setattr(market_news, "get_exchange_market_close", lambda _: None)
    feed = market_news.fetch_recent_news("SHEL.L")
    assert feed["items"][0]["id"] == "newest"
    assert market_news.summarize_cached_news_context("SHEL.L")["latest_published_at"] == later


def test_latest_timestamp_includes_filings_and_skips_invalid_dates() -> None:
    with market_news._lock:
        market_news._cache["MSFT"] = {
            "items": [{"published_at": "unknown"}, {"published_at": "2026-11-24T19:00:00+01:00"}],
            "sec_filings": [{"id": "sec-1", "published_at": "2026-11-24T18:30:00Z"}],
        }
    assert (
        market_news.summarize_cached_news_context("MSFT")["latest_published_at"]
        == "2026-11-24T18:30:00Z"
    )


def test_route_total_time_includes_news_context(monkeypatch) -> None:
    now = 100.0

    def read_news(_: str) -> dict:
        nonlocal now
        now += 0.125
        return {}

    monkeypatch.setattr(forecast_route, "time", SimpleNamespace(perf_counter=lambda: now))
    monkeypatch.setattr(forecast_route, "_download_ohlcv", lambda _: pd.DataFrame())
    monkeypatch.setattr(forecast_route, "train_and_forecast", lambda *_: {})
    monkeypatch.setattr(forecast_route, "summarize_cached_news_context", read_news)
    response = CLIENT.get("/api/v1/forecast?ticker=MSFT")
    assert response.status_code == 200
    assert response.json()["timing"]["total_ms"] == 125
    assert "total;dur=125" in response.headers["Server-Timing"]


def test_forecast_route_attaches_news_context_without_altering_conformal_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _synthetic_ohlcv(520)
    monkeypatch.setattr(forecast_route, "_download_ohlcv", lambda symbol, **kw: df)

    # Cold news cache -> non-blocking not_cached context
    resp_cold = CLIENT.get("/api/v1/forecast?ticker=MSFT")
    assert resp_cold.status_code == 200
    body_cold = resp_cold.json()
    assert body_cold["news_context"]["status"] == "not_cached"
    assert body_cold["interval_calibration"]["m5_gate_verdict"] == "PASS"

    # Warm news cache -> aggregated bullish context attached; predictions remain bit-for-bit identical
    with market_news._lock:
        market_news._cache["MSFT"] = {
            "status": "available",
            "items": [
                {
                    "id": "a1",
                    "title": "Microsoft cloud revenue surges",
                    "published_at": "2026-11-25T14:00:00Z",
                    "sentiment": 0.55,
                    "sentiment_label": "bullish",
                    "kind": "news",
                },
                {
                    "id": "a2",
                    "title": "Azure enterprise demand rises",
                    "published_at": "2026-11-24T12:00:00Z",
                    "sentiment": 0.25,
                    "sentiment_label": "bullish",
                    "kind": "news",
                },
            ],
            "articles": [],
            "sec_filings": [{"id": "s1", "form": "8-K", "published_at": "2026-11-23T20:00:00Z"}],
            "provider": "alpaca",
            "as_of": "2026-11-25T18:00:00Z",
        }

    resp_warm = CLIENT.get("/api/v1/forecast?ticker=MSFT")
    assert resp_warm.status_code == 200
    body_warm = resp_warm.json()
    assert body_warm["news_context"]["status"] == "available"
    assert body_warm["news_context"]["sentiment_label"] == "bullish"
    assert body_warm["news_context"]["mean_compound"] == pytest.approx(0.4, abs=1e-4)
    assert body_warm["news_context"]["headline_count"] == 2
    assert body_warm["news_context"]["sec_filing_count"] == 1
    assert body_warm["news_context"]["provider"] == "alpaca"
    assert body_warm["news_context"]["used_in_point_model"] is False
    # Model outputs remain strictly unchanged (M5 calibration gate preserved)
    assert body_warm["predicted_prices"] == body_cold["predicted_prices"]
    assert body_warm["lower_prices"] == body_cold["lower_prices"]
    assert body_warm["upper_prices"] == body_cold["upper_prices"]
    assert body_warm["interval_calibration"] == body_cold["interval_calibration"]
