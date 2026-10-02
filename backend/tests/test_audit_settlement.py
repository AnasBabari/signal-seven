"""Audit regressions: authenticated settlement, units and immutable evidence."""

import math

import httpx
import pandas as pd
from fastapi.testclient import TestClient

from api import app
from routes import volatility
from scripts import collect_live_forecasts as collector
from services.forecast_ledger import ForecastLedger, LedgerUnavailableError


def test_authenticated_settlement_mature_pending_and_missing_origin(monkeypatch, tmp_path):
    ledger = ForecastLedger(tmp_path / "ledger.db")
    dates = pd.bdate_range("2026-09-01", periods=12)
    prices = [100 + i for i in range(12)]
    frame = pd.DataFrame({"Close": prices}, index=dates)
    for origin in [dates[0].date().isoformat(), dates[-2].date().isoformat(), "2026-08-01"]:
        ledger.record_forecast(
            forecast_date=origin,
            ticker="AAPL",
            horizon=5,
            target_date="2026-10-01",
            model_name="rolling_mean",
            predicted_volatility=0.2,
            recent_realized_volatility=0.18,
            origin_price=100,
            lower_scenario_price=90,
            upper_scenario_price=110,
            model_version="deployable_v5:annualized_sigma_v1",
        )
    monkeypatch.setenv("FORECAST_COLLECTOR_TOKEN", "test-secret")
    monkeypatch.setattr(volatility, "get_forecast_ledger", lambda: ledger)
    monkeypatch.setattr(volatility, "_download_ohlcv", lambda symbol: frame)
    client = TestClient(app)
    assert client.post("/api/v1/volatility/score-ledger?ticker=AAPL").status_code == 401
    response = client.post(
        "/api/v1/volatility/score-ledger?ticker=AAPL",
        headers={"Authorization": "Bearer test-secret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["scored_count"] == 1
    assert body["live_track_record"]["total_forecasts"] == 3
    assert body["live_track_record"]["pending_forecasts"] == 2
    assert body["live_track_record"]["scored_forecasts"] == 1
    assert body["replay_track_record"]["total_forecasts"] == 0
    assert ledger.score_pending_forecasts("AAPL", frame) == 0


def test_settlement_storage_failure_is_sanitized_without_data_or_fallback(monkeypatch):
    monkeypatch.setenv("FORECAST_COLLECTOR_TOKEN", "test-secret")

    def fail():
        raise LedgerUnavailableError("postgres://private-credential")

    monkeypatch.setattr(volatility, "get_forecast_ledger", fail)
    monkeypatch.setattr(
        volatility,
        "_download_ohlcv",
        lambda symbol: (_ for _ in ()).throw(AssertionError("must not download")),
    )
    response = TestClient(app).post(
        "/api/v1/volatility/score-ledger?ticker=AAPL",
        headers={"Authorization": "Bearer test-secret"},
    )
    assert response.status_code == 503
    assert "private-credential" not in response.text


def test_recent_variance_is_converted_to_annualized_sigma(monkeypatch):
    from test_volatility_baseline_route import _snapshot

    monkeypatch.setattr(volatility, "build_volatility_inference_snapshot", _snapshot)
    prepared = volatility._prepare_forecast("MSFT", 5, "rolling_mean")
    assert prepared.record_kwargs["recent_realized_volatility"] == math.sqrt(252 * 0.04)
    assert prepared.record_kwargs["model_version"].endswith(":annualized_sigma_v1")


def test_retry_after_and_resumable_settlement_use_virtual_delays(monkeypatch):
    delays = []
    monkeypatch.setattr(collector.time, "sleep", delays.append)
    client = collector.CollectorClient("https://example.test")
    replies = iter(
        [
            httpx.Response(429, headers={"Retry-After": "7"}, json={}),
            httpx.Response(200, json={"scored_count": 1}),
        ]
    )
    monkeypatch.setattr(client._client, "request", lambda *args, **kwargs: next(replies))
    assert client.request("POST", "/score").status_code == 200
    assert delays == [7]
    calls = []
    monkeypatch.setattr(
        client,
        "request",
        lambda *args, **kwargs: (
            calls.append(kwargs["params"]["ticker"])
            or collector.ApiResult(200, {"scored_count": 0})
        ),
    )
    first = collector.LIVE_UNIVERSE_V1[0]
    result = collector.run_settlement(
        client,
        token="test",
        previous={"items": [{"ticker": first, "status": "succeeded", "scored_count": 1}]},
    )
    assert result["status"] == "complete"
    assert first not in calls
    assert all(delay == 6.1 for delay in delays[1:])
    client.close()
