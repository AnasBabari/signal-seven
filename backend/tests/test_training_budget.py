from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app
from config import settings
from services import simple_forecast as sf
from services.training_budget import TrainingBusyError, training_slot


def test_different_tickers_share_capacity_but_memory_hits_still_work(monkeypatch):
    monkeypatch.setattr(settings, "forecast_training_max_concurrency", 1)
    monkeypatch.setattr(settings, "forecast_training_coordination_database_url", None)
    monkeypatch.setattr(sf, "_load_fitted_forecast", lambda key: None)
    frame = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2026-10-01"]))
    sf.clear_forecast_cache()
    fitted = []
    monkeypatch.setattr(
        sf, "_fit_forecast", lambda *args: fitted.append(args[0]) or {"ticker": args[0]}
    )
    with training_slot(), ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(TrainingBusyError):
            pool.submit(sf.train_and_forecast, "MSFT", frame, "ridge").result(timeout=3)
        with sf._cache_lock:
            sf._cache[sf._forecast_cache_key("AAPL", frame, "ridge")] = {
                "ticker": "AAPL",
                "predicted_prices": [101.0],
            }
        result = pool.submit(sf.train_and_forecast, "AAPL", frame, "ridge").result(timeout=3)
        assert result["timing"]["cache_status"] == "memory_hit"
    assert fitted == []
    assert sf.train_and_forecast("MSFT", frame, "ridge")["ticker"] == "MSFT"
    sf.clear_forecast_cache()


def test_slot_is_released_when_fitting_fails(monkeypatch):
    monkeypatch.setattr(settings, "forecast_training_coordination_database_url", None)
    with pytest.raises(ValueError), training_slot():
        raise ValueError("failed fit")
    with training_slot():
        pass


@pytest.mark.parametrize("acquired", [True, False])
def test_optional_postgres_coordination_is_bounded_and_releases_session(monkeypatch, acquired):
    import psycopg

    monkeypatch.setattr(
        settings, "forecast_training_coordination_database_url", "postgresql://fixture"
    )
    queries, closed = [], []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql, args=None):
            queries.append((sql, args))

        def fetchone(self):
            return [acquired]

    def connect(url, **kwargs):
        assert kwargs == {"connect_timeout": 2, "autocommit": True}
        return SimpleNamespace(cursor=Cursor, close=lambda: closed.append(True))

    monkeypatch.setattr(psycopg, "connect", connect)
    if acquired:
        with training_slot():
            pass
    else:
        with pytest.raises(TrainingBusyError), training_slot():
            pass
    assert closed == [True]
    assert "statement_timeout" in queries[0][0]
    assert "pg_try_advisory_lock" in queries[1][0]


def test_forecast_route_returns_retryable_busy_without_placeholder(monkeypatch):
    from routes import simple_forecast

    monkeypatch.setattr(simple_forecast, "_download_ohlcv", lambda ticker: None)

    def busy(*args):
        raise TrainingBusyError("Forecast training is busy. Please retry shortly.")

    monkeypatch.setattr(simple_forecast, "train_and_forecast", busy)
    response = TestClient(app).get("/api/v1/forecast?ticker=MSFT")
    assert response.status_code == 503
    assert response.headers["retry-after"] == "3"
    assert "predicted_prices" not in response.json()


def test_discovery_matches_actual_universe_and_evidence():
    body = TestClient(app).get("/models").json()
    price = body["simple_price_forecast"]
    assert set(price["supported_tickers"]) == set(sf.SUPPORTED_TICKERS)
    assert price["supported_ticker_count"] == len(sf.SUPPORTED_TICKERS) == 286
    assert price["evaluated_series"] == "predicted_prices"
    assert price["midpoint_evaluation"] == "unavailable"
