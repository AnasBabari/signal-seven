"""Milestone 03 backend performance, parity, caching, and resource-safety tests."""

from __future__ import annotations

import gzip
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import psutil
import pytest
from fastapi.testclient import TestClient

from api import app
from config import settings
from market_data.alpaca import AlpacaProvider
from market_data.base import MarketDataResult
from market_data.cache import MEMORY_CACHE_MAXSIZE, MarketDataCache
from market_data.normalization import daily_frame_fingerprint
from market_data.service import MarketDataService
from routes import market, volatility
from services import forecast_ledger, g3_volatility, simple_forecast, volatility_snapshot
from services.forecast_warmup import DEFAULT_WARMUP_TICKERS
from services.live_volatility import build_live_volatility_forecast
from services.training_budget import training_slot

CLIENT = TestClient(app)


def _reference_causal_log_har_forecasts(
    rv_daily: np.ndarray,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    *,
    minimum_history: int = 60,
    refit_every: int = 5,
    ridge: float = 1e-4,
) -> np.ndarray:
    """Exact pre-M3 reference loop kept solely as an oracle for parity verification."""
    rv = np.asarray(rv_daily, dtype=np.float64)
    output = np.full((len(rv), len(horizons)), np.nan, dtype=np.float64)
    coefficients: np.ndarray | None = None
    last_refit = -refit_every
    maximum_horizon = max(horizons)
    horizon_to_column = {horizon: column for column, horizon in enumerate(horizons)}
    xtx = np.zeros((4, 4), dtype=np.float64)
    xty = np.zeros(4, dtype=np.float64)
    fitted_rows = 0
    penalty = np.eye(4, dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0

    for origin in range(22, len(rv)):
        training_origin = origin - 1
        if np.isfinite(rv[training_origin - 21 : training_origin + 2]).all():
            design = volatility_snapshot._log_har_row(
                rv[training_origin - 21 : training_origin + 1]
            )
            response = float(np.log(max(rv[origin], 1e-12)))
            xtx += np.outer(design, design)
            xty += design * response
            fitted_rows += 1
        if origin < max(minimum_history, 22) or fitted_rows < 20:
            continue
        if not np.isfinite(rv[origin - 21 : origin + 1]).all():
            continue
        if coefficients is None or origin - last_refit >= refit_every:
            coefficients = np.linalg.solve(xtx + penalty, xty)
            last_refit = origin

        history_tail = list(np.maximum(rv[origin - 21 : origin + 1], 1e-12))
        cumulative = 0.0
        for step in range(1, maximum_horizon + 1):
            row = volatility_snapshot._log_har_row(np.asarray(history_tail, dtype=np.float64))
            next_variance = float(np.exp(np.clip(row @ coefficients, -30.0, 5.0)))
            next_variance = max(next_variance, 1e-12)
            history_tail.append(next_variance)
            del history_tail[0]
            cumulative += next_variance
            if step in horizon_to_column:
                output[origin, horizon_to_column[step]] = cumulative
    return output


def _synthetic_ohlcv(rows: int = 750, *, seed: int = 17, end: str = "2026-09-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.013, size=rows)
    close = 120.0 * np.exp(np.cumsum(returns))
    open_ = close * np.exp(rng.normal(0.0, 0.0025, size=rows))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.014, size=rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.014, size=rows))
    frame = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(500_000, 5_000_000, size=rows).astype(float),
        },
        index=pd.bdate_range(end=end, periods=rows),
    )
    frame.attrs.update(
        {
            "data_provider": "alpaca",
            "data_as_of": frame.index[-1].date().isoformat(),
            "data_feed": "iex",
            "price_adjustment": "all",
            "data_fingerprint": daily_frame_fingerprint(frame),
            "market_data_cache": "hit",
        }
    )
    return frame


@pytest.mark.parametrize("length", [120, 121, 122, 123, 124, 500, 501, 502, 503, 504])
def test_vectorized_har_matches_reference_across_all_refit_remainders(length: int) -> None:
    rng = np.random.default_rng(1000 + length)
    rv = np.exp(rng.normal(-8.5, 0.9, size=length))
    horizons = volatility_snapshot.VOLATILITY_PATH_HORIZONS

    expected = _reference_causal_log_har_forecasts(rv, horizons, refit_every=5)
    got_full = volatility_snapshot.causal_log_har_forecasts(rv, horizons, refit_every=5)
    got_latest = volatility_snapshot.causal_log_har_forecasts(
        rv, horizons, refit_every=5, latest_only=True
    )

    np.testing.assert_allclose(got_full, expected, rtol=1e-6, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(got_latest[-1], expected[-1], rtol=1e-6, atol=0.0, equal_nan=True)


def test_vectorized_har_matches_reference_with_nans() -> None:
    rng = np.random.default_rng(77)
    rv = np.exp(rng.normal(-8.0, 0.8, size=213))
    rv[35] = np.nan
    rv[91] = np.nan
    rv[140] = np.nan

    expected = _reference_causal_log_har_forecasts(rv, (1, 5, 10, 20), refit_every=5)
    got_full = volatility_snapshot.causal_log_har_forecasts(rv, (1, 5, 10, 20), refit_every=5)
    got_latest = volatility_snapshot.causal_log_har_forecasts(
        rv, (1, 5, 10, 20), refit_every=5, latest_only=True
    )

    np.testing.assert_allclose(got_full, expected, rtol=1e-6, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(got_latest[-1], expected[-1], rtol=1e-6, atol=0.0, equal_nan=True)


def test_lazy_garch_is_skipped_on_h5_and_exact_on_h1(monkeypatch) -> None:
    volatility_snapshot.clear_snapshot_cache()
    frame = _synthetic_ohlcv(520)
    monkeypatch.setattr(
        volatility_snapshot,
        "_download_ohlcv",
        lambda _ticker, **_kwargs: frame.copy(),
    )

    garch_calls = {"count": 0}
    real_garch = volatility_snapshot._garch11_cumulative_variance_path

    def spy_garch(close, maximum_horizon=20):
        garch_calls["count"] += 1
        return real_garch(close, maximum_horizon)

    monkeypatch.setattr(volatility_snapshot, "_garch11_cumulative_variance_path", spy_garch)

    snapshot = volatility_snapshot.build_volatility_inference_snapshot("MSFT")
    assert garch_calls["count"] == 0
    assert snapshot.garch_variance_path is None
    assert "garch_11" not in snapshot.baseline_candidates
    assert isinstance(snapshot.raw_ohlcv, np.ndarray)
    assert not snapshot.raw_ohlcv.flags.writeable

    # H=5 auto (rolling_mean) must still NOT invoke GARCH(1,1).
    h5 = build_live_volatility_forecast(snapshot, horizon=5, model="auto")
    assert h5["forecast"]["model"] == "rolling_mean"
    assert garch_calls["count"] == 0

    # H=1 auto routes to garch_11 and computes lazily ONCE, matching exact MLE path.
    h1_first = build_live_volatility_forecast(snapshot, horizon=1, model="auto")
    h1_second = build_live_volatility_forecast(snapshot, horizon=1, model="garch_11")
    assert h1_first["forecast"]["model"] == "garch_11"
    assert garch_calls["count"] == 1
    direct_path = real_garch(frame["Close"].to_numpy(dtype=np.float64), 20)
    assert h1_first["forecast"]["expected_cumulative_variance"] == pytest.approx(
        float(direct_path[0]), rel=1e-12
    )
    assert h1_second["forecast"]["expected_cumulative_variance"] == pytest.approx(
        float(direct_path[0]), rel=1e-12
    )

    # _snapshot_identity ignores garch_11 in baseline_paths.
    with_garch = {**(snapshot.baseline_variance_paths or {}), "garch_11": direct_path}
    id_without = volatility_snapshot._snapshot_identity(
        "MSFT",
        frame.index[-60:],
        snapshot.features,
        snapshot.causal_har_variance,
        snapshot.baseline_variance_paths,
        data_provider="alpaca",
        data_as_of="2026-09-02",
    )
    id_with = volatility_snapshot._snapshot_identity(
        "MSFT",
        frame.index[-60:],
        snapshot.features,
        snapshot.causal_har_variance,
        with_garch,
        data_provider="alpaca",
        data_as_of="2026-09-02",
    )
    assert id_without == id_with
    volatility_snapshot.clear_snapshot_cache()


def test_volatility_batch_parity_single_pass_g3_and_partial_errors(monkeypatch) -> None:
    volatility_snapshot.clear_snapshot_cache()
    frame = _synthetic_ohlcv(520)
    monkeypatch.setattr(
        volatility_snapshot,
        "_download_ohlcv",
        lambda _ticker, **_kwargs: frame.copy(),
    )

    # 1. Parity for model=auto across horizons=5,10,20
    batch_resp = CLIENT.get(
        "/api/v1/volatility/forecast/batch",
        params={"ticker": "MSFT", "horizons": "5,10,20", "model": "auto"},
    )
    assert batch_resp.status_code == 200
    batch = batch_resp.json()
    assert batch["ticker"] == "MSFT"
    assert batch["horizons"] == [5, 10, 20]
    assert batch["errors"] == {}
    assert batch["post_close_grace_used"] is False
    for horizon in (5, 10, 20):
        single = CLIENT.get(
            "/api/v1/volatility/forecast",
            params={"ticker": "MSFT", "horizon": horizon, "model": "auto"},
        )
        assert single.status_code == 200
        assert batch["forecasts"][str(horizon)] == single.json()

    # 2. Single-pass G3 features across horizons=5,10,20
    g3_feature_calls = {"count": 0}
    real_build_g3 = g3_volatility.build_g3_features

    def spy_g3_features(df):
        g3_feature_calls["count"] += 1
        return real_build_g3(df)

    monkeypatch.setattr(g3_volatility, "build_g3_features", spy_g3_features)
    g3_batch = CLIENT.get(
        "/api/v1/volatility/forecast/batch",
        params={"ticker": "MSFT", "horizons": "5,10,20", "model": "gpu_g3"},
    )
    assert g3_batch.status_code == 200
    assert g3_feature_calls["count"] == 1
    g3_body = g3_batch.json()
    for horizon in (5, 10, 20):
        single_g3 = CLIENT.get(
            "/api/v1/volatility/forecast",
            params={"ticker": "MSFT", "horizon": horizon, "model": "gpu_g3"},
        )
        assert single_g3.status_code == 200
        assert g3_body["forecasts"][str(horizon)] == single_g3.json()

    # 3. Per-horizon error handling preserves successful horizons in partial failure
    real_prepare = volatility._prepare_forecast_from_snapshot

    def flaky_prepare(snapshot, symbol, horizon, requested_model, *, g3_context=None):
        if horizon == 10:
            raise ValueError("synthetic horizon 10 failure")
        return real_prepare(snapshot, symbol, horizon, requested_model, g3_context=g3_context)

    monkeypatch.setattr(volatility, "_prepare_forecast_from_snapshot", flaky_prepare)
    partial = CLIENT.get(
        "/api/v1/volatility/forecast/batch",
        params={"ticker": "MSFT", "horizons": "5,10,20", "model": "auto"},
    )
    assert partial.status_code == 200
    partial_body = partial.json()
    assert set(partial_body["forecasts"].keys()) == {"5", "20"}
    assert "10" in partial_body["errors"]
    assert partial_body["errors"]["10"]["code"] == "HORIZON_UNAVAILABLE"

    # 4. Invalid horizons -> 400
    assert CLIENT.get("/api/v1/volatility/forecast/batch?horizons=5,7").status_code == 400
    assert CLIENT.get("/api/v1/volatility/forecast/batch?horizons=abc").status_code == 400
    volatility_snapshot.clear_snapshot_cache()


def test_onnx_session_options_pin_single_thread(monkeypatch) -> None:
    import onnxruntime as ort

    captured: dict[str, object] = {}
    real_cls = ort.InferenceSession

    def spy_session(path, sess_options=None, providers=None, **kwargs):
        captured["intra"] = sess_options.intra_op_num_threads if sess_options else None
        captured["inter"] = sess_options.inter_op_num_threads if sess_options else None
        return real_cls(path, sess_options=sess_options, providers=providers, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", spy_session)
    monkeypatch.setattr(g3_volatility, "_sessions", {})
    g3_volatility._load_session(5)
    assert captured["intra"] == 1
    assert captured["inter"] == 1


def test_history_etag_304_cache_control_grace_and_gzip(monkeypatch) -> None:
    frame = _synthetic_ohlcv(1500)
    fp = frame.attrs["data_fingerprint"]
    normal_result = MarketDataResult(
        frame=frame,
        provider="alpaca",
        data_as_of=frame.index[-1].date().isoformat(),
        cache_status="hit",
        data_fingerprint=fp,
        post_close_grace_used=False,
    )
    monkeypatch.setattr(market, "_fetch_history_daily", lambda _symbol: normal_result)
    monkeypatch.setattr(market, "_fetch_history_intraday", lambda _symbol: None)

    resp = CLIENT.get(
        "/api/v1/history",
        params={"ticker": "AAPL"},
        headers={"Accept-Encoding": "gzip"},
    )
    assert resp.status_code == 200
    expected_etag = f'W/"{fp}"'
    assert resp.headers.get("etag") == expected_etag
    assert resp.headers.get("cache-control") == "public, max-age=60, stale-while-revalidate=300"
    assert resp.headers.get("content-encoding") == "gzip"
    assert len(resp.json()["daily"]) == 1500
    assert len(gzip.compress(resp.content)) > 0  # decompressed cleanly by TestClient

    # Weak & strong & list If-None-Match -> 304 Not Modified
    for header_val in (expected_etag, f'"{fp}"', f'W/"other", {expected_etag}'):
        not_mod = CLIENT.get(
            "/api/v1/history",
            params={"ticker": "AAPL"},
            headers={"If-None-Match": header_val},
        )
        assert not_mod.status_code == 304
        assert not_mod.content == b""
        assert not_mod.headers.get("etag") == expected_etag

    # Post-close grace fallback must emit Cache-Control: no-store and skip ETag/304.
    grace_result = MarketDataResult(
        frame=frame,
        provider="alpaca",
        data_as_of=frame.index[-1].date().isoformat(),
        cache_status="miss",
        data_fingerprint=fp,
        post_close_grace_used=True,
    )
    monkeypatch.setattr(market, "_fetch_history_daily", lambda _symbol: grace_result)
    grace_resp = CLIENT.get(
        "/api/v1/history",
        params={"ticker": "AAPL"},
        headers={"If-None-Match": expected_etag},
    )
    assert grace_resp.status_code == 200
    assert grace_resp.headers.get("cache-control") == "no-store"
    assert "etag" not in grace_resp.headers


def test_market_data_cache_lru_shallow_copy_and_o1_fresh_count(monkeypatch, tmp_path) -> None:
    cache = MarketDataCache(tmp_path, memory_maxsize=4)
    assert MEMORY_CACHE_MAXSIZE == 64

    frame = _synthetic_ohlcv(100, end="2026-09-02")
    us_res = MarketDataResult(
        frame=frame,
        provider="alpaca",
        data_as_of="2026-09-02",
        data_fingerprint=frame.attrs["data_fingerprint"],
    )
    lse_res = MarketDataResult(
        frame=frame,
        provider="yahoo",
        data_as_of="2026-09-02",
        data_fingerprint=frame.attrs["data_fingerprint"],
    )
    cache.save("AAPL", us_res)
    cache.save("MSFT", us_res)
    cache.save("SHEL.L", lse_res)

    # Steady-state load and fresh_entry_count must NOT read files from disk.
    real_read_text = Path.read_text

    def forbid_disk_read(self, *args, **kwargs):
        if self.parent == tmp_path:
            raise AssertionError("Steady-state memory hit / O(1) fresh_entry_count read disk!")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", forbid_disk_read)

    hit1 = cache.load("alpaca", "AAPL", required_session="2026-09-02")
    hit2 = cache.load("alpaca", "AAPL", required_session="2026-09-02")
    assert hit1 is not None and hit2 is not None
    # Shallow copy shares underlying values while isolating frame wrapper & attrs.
    assert hit1.frame is not hit2.frame
    hit1.frame.attrs["caller_tag"] = "mutated"
    assert "caller_tag" not in hit2.frame.attrs

    assert cache.fresh_entry_count(required_session="2026-09-02") == 3
    assert (
        cache.fresh_entry_count(
            required_session="2026-09-02", providers={"alpaca"}, exclude_lse_symbols=True
        )
        == 2
    )
    assert (
        cache.fresh_entry_count(
            required_session="2026-09-02", providers={"yahoo"}, exclude_lse_symbols=True
        )
        == 0
    )
    assert (
        cache.fresh_entry_count(
            required_session="2026-09-02", providers={"yahoo"}, exclude_lse_symbols=False
        )
        == 1
    )

    # LRU eviction bounds in-memory size to memory_maxsize=4.
    monkeypatch.undo()
    for sym in ("NVDA", "GOOGL", "AMZN"):
        cache.save(sym, us_res)
    assert len(cache._memory_cache) == 4


def test_alpaca_provider_pooled_client_reuse_and_close() -> None:
    provider = AlpacaProvider(key_id="k", secret_key="s")
    c1 = provider._get_client()
    c2 = provider._get_client()
    assert c1 is c2
    assert not c1.is_closed
    provider.close()
    assert c1.is_closed

    # Injected test client must never be closed by AlpacaProvider.close().
    injected = httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={})))
    p_injected = AlpacaProvider(key_id="k", secret_key="s", client=injected)
    assert p_injected._get_client() is injected
    p_injected.close()
    assert not injected.is_closed
    injected.close()


def test_calendar_cache_ttl_bypass_and_market_news_reuse(monkeypatch) -> None:
    import calendars
    from services.market_news import get_exchange_market_close

    calendars.clear_calendar_caches()
    first = calendars.completed_session_context(exchange="NYSE")
    second = calendars.completed_session_context(exchange="NYSE")
    assert first is second

    # Explicit `now` never uses or poisons the live TTL cache.
    historical = calendars.completed_session_context(
        now=datetime(2026, 9, 4, 21, 0, tzinfo=UTC), exchange="NYSE"
    )
    assert historical.required_session == pd.Timestamp("2026-09-04")
    assert calendars.completed_session_context(exchange="NYSE") is first

    # bypass_session_cache forces fresh evaluation without returning the cached object.
    with calendars.bypass_session_cache():
        fresh = calendars.completed_session_context(exchange="NYSE")
    assert fresh is not first
    assert fresh.required_session == first.required_session

    # market_news uses shared calendar cache without direct mcal.get_calendar instantiation.
    close_dt = get_exchange_market_close("AAPL")
    assert close_dt is not None
    assert close_dt == first.market_close_utc.to_pydatetime().astimezone(UTC)


def test_deterministic_ridge_auto_serving_opt_in_rf_and_budget_timeout(monkeypatch) -> None:
    simple_forecast.clear_forecast_cache()
    assert simple_forecast._cache.maxsize == 128
    assert DEFAULT_WARMUP_TICKERS == ("AAPL", "MSFT")
    assert settings.forecast_auto_candidates == "ridge"

    frame = _synthetic_ohlcv(700)
    auto_res = simple_forecast.train_and_forecast("AAPL", frame, "auto")
    assert auto_res["model"]["selected"] == "ridge"
    assert list(auto_res["model"]["candidate_validation_mae"].keys()) == ["ridge"]

    # Explicit opt-in random_forest remains fully supported.
    rf_res = simple_forecast.train_and_forecast("AAPL", frame, "random_forest")
    assert rf_res["model"]["selected"] == "random_forest"
    assert list(rf_res["model"]["candidate_validation_mae"].keys()) == ["random_forest"]

    # Flipping FORECAST_AUTO_CANDIDATES invalidates cache key and evaluates configured candidates.
    key_ridge = simple_forecast._forecast_cache_key("AAPL", frame, "auto")
    monkeypatch.setattr(settings, "forecast_auto_candidates", "ridge,random_forest")
    key_both = simple_forecast._forecast_cache_key("AAPL", frame, "auto")
    assert key_both != key_ridge

    # Brief contention within forecast_training_acquire_timeout_seconds succeeds without 503.
    monkeypatch.setattr(settings, "forecast_training_max_concurrency", 1)
    monkeypatch.setattr(settings, "forecast_training_acquire_timeout_seconds", 1.5)
    monkeypatch.setattr(settings, "forecast_training_coordination_database_url", None)

    entered = threading.Event()
    finished = []

    def brief_holder():
        with training_slot():
            entered.set()
            time.sleep(0.08)

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(brief_holder)
        assert entered.wait(timeout=2.0)
        with training_slot():
            finished.append(True)
        fut.result(timeout=2.0)
    assert finished == [True]
    simple_forecast.clear_forecast_cache()


def test_code_commit_lru_cache_prefers_render_git_commit(monkeypatch) -> None:
    forecast_ledger.get_current_code_commit.cache_clear()
    monkeypatch.setenv("RENDER_GIT_COMMIT", "deadbeef12345678")
    monkeypatch.setattr(
        forecast_ledger.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("git subprocess must not run")),
    )
    assert forecast_ledger.get_current_code_commit() == "deadbeef12345678"
    # Second call served from lru_cache even if env changes.
    monkeypatch.setenv("RENDER_GIT_COMMIT", "other")
    assert forecast_ledger.get_current_code_commit() == "deadbeef12345678"
    forecast_ledger.get_current_code_commit.cache_clear()


def test_lifespan_shutdown_closes_alpaca_and_ledger_pools(monkeypatch) -> None:
    import api
    import data_pipeline
    from services import forecast_ledger as fl

    closed = {"market_data": 0, "ledger": 0}
    monkeypatch.setattr(
        data_pipeline.market_data_service,
        "close",
        lambda: closed.__setitem__("market_data", closed["market_data"] + 1),
    )
    monkeypatch.setattr(
        fl,
        "close_forecast_ledger",
        lambda: closed.__setitem__("ledger", closed["ledger"] + 1),
    )
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
    assert closed == {"market_data": 1, "ledger": 1}


def test_volatility_ledger_route_reuses_one_connection(monkeypatch, tmp_path) -> None:
    ledger = forecast_ledger.ForecastLedger(tmp_path / "single_conn.db")
    conn_opens = {"count": 0}
    real_get_conn = ledger._get_connection

    def counting_get_conn():
        conn_opens["count"] += 1
        return real_get_conn()

    monkeypatch.setattr(ledger, "_get_connection", counting_get_conn)
    monkeypatch.setattr(volatility, "get_forecast_ledger", lambda: ledger)

    resp = CLIENT.get("/api/v1/volatility/ledger", params={"ticker": "MSFT", "horizon": 5})
    assert resp.status_code == 200
    assert conn_opens["count"] == 1


def test_warm_cache_hit_micro_benchmark_and_rss_bound(monkeypatch, tmp_path) -> None:
    """Prove warm cache hits stay well under 350ms and memory remains bounded for 512MB tier."""
    import data_pipeline
    from calendars import latest_completed_trading_session
    from routes import health
    from routes.common import limiter

    monkeypatch.setattr(limiter, "enabled", False)
    required = latest_completed_trading_session().date().isoformat()
    frame = _synthetic_ohlcv(1600, end=required)
    provider = SimpleNamespace(
        name="alpaca",
        configured=True,
        fetch_daily_bars=lambda _sym, *, years: MarketDataResult(
            frame=frame.copy(),
            provider="alpaca",
            data_as_of=required,
            data_fingerprint=frame.attrs["data_fingerprint"],
        ),
    )
    service = MarketDataService([provider], cache=MarketDataCache(tmp_path / "bench_cache"))
    monkeypatch.setattr(data_pipeline, "market_data_service", service)
    monkeypatch.setattr(health, "market_data_service", service)
    monkeypatch.setattr(
        health,
        "market_circuit_breaker",
        SimpleNamespace(is_ready=lambda: (True, {"status": "available", "circuit": "closed"})),
    )
    monkeypatch.setattr(market, "_fetch_history_intraday", lambda _s: None)
    volatility_snapshot.clear_snapshot_cache()
    simple_forecast.clear_forecast_cache()

    # Cold warm-up call for each endpoint
    assert CLIENT.get("/api/v1/history?ticker=AAPL").status_code == 200
    assert (
        CLIENT.get("/api/v1/volatility/forecast/batch?ticker=AAPL&horizons=5,10,20").status_code
        == 200
    )
    assert CLIENT.get("/api/v1/forecast?ticker=AAPL").status_code == 200

    # Measure 20 warm sequential hits per endpoint
    timings: dict[str, list[float]] = {"history": [], "vol_batch": [], "forecast": [], "ready": []}
    for _ in range(20):
        for name, url in (
            ("history", "/api/v1/history?ticker=AAPL"),
            ("vol_batch", "/api/v1/volatility/forecast/batch?ticker=AAPL&horizons=5,10,20"),
            ("forecast", "/api/v1/forecast?ticker=AAPL"),
            ("ready", "/ready"),
        ):
            t0 = time.perf_counter()
            resp = CLIENT.get(url)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert resp.status_code == 200
            timings[name].append(elapsed_ms)

    for name, samples in timings.items():
        p50 = float(np.median(samples))
        assert p50 < 350.0, f"Warm {name} p50={p50:.1f}ms exceeded 350ms SLO"

    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    assert rss_mb < 512.0, f"Process RSS {rss_mb:.1f}MB exceeded 512MB Render Free envelope"
    volatility_snapshot.clear_snapshot_cache()
    simple_forecast.clear_forecast_cache()
