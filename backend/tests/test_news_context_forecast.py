"""TDD tests for non-blocking aggregated news sentiment context on /api/v1/forecast."""

from __future__ import annotations

import time
from typing import Any

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
    ("compounds", "expected_label"),
    [
        ([0.42, 0.28, 0.15], "bullish"),
        ([-0.38, -0.22, -0.05], "bearish"),
        ([0.12, -0.10, 0.01], "neutral"),
    ],
)
def test_summarize_cached_news_context_aggregates_sentiment_and_sec_filings(
    compounds: list[float],
    expected_label: str,
) -> None:
    items = [
        {
            "id": f"n-{i}",
            "title": f"Headline {i}",
            "published_at": f"2026-11-2{i}T16:00:00Z",
            "sentiment": c,
            "sentiment_label": "bullish" if c >= 0.08 else ("bearish" if c <= -0.08 else "neutral"),
            "kind": "news",
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
    assert summary["mean_compound"] == pytest.approx(sum(compounds) / len(compounds), abs=1e-4)
    assert summary["headline_count"] == len(compounds)
    assert summary["sec_filing_count"] == 1
    assert summary["provider"] == "yahoo"
    assert summary["latest_published_at"] == "2026-11-24T21:00:00Z"
    assert summary["used_in_point_model"] is False


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
