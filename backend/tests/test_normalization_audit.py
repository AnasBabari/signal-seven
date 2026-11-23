from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app
from calendars import completed_session_context
from market_data.base import (
    BarRepairRecord,
    MarketDataProviderError,
    MarketDataResult,
    MarketDataServiceError,
)
from market_data.cache import CACHE_SCHEMA_VERSION, MarketDataCache
from market_data.normalization import daily_frame_fingerprint, normalize_daily_bars_with_audit
from market_data.service import MarketDataService
from market_data.yahoo import YahooProvider
from services.simple_forecast import get_ticker_meta

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "yahoo_lse_anomalies.json"
CLIENT = TestClient(app)


def _load_fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _synthetic_frame(rows: int = 2020, end: str = "2026-08-31") -> pd.DataFrame:
    index = pd.bdate_range(end=end, periods=rows)
    close = np.full(rows, 500.0, dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 1.0,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": np.full(rows, 1_000_000.0, dtype=float),
        },
        index=index,
    )


def test_yahoo_lse_fixture_repairs_produce_audit_trail_and_logs(caplog) -> None:
    fixtures = _load_fixtures()["fixtures"]

    # SHEL.L: single High < Open repair
    shel = _synthetic_frame(rows=fixtures["SHEL.L"]["rows"])
    shel_anom = fixtures["SHEL.L"]["anomalies"][0]
    ts = pd.Timestamp(shel_anom["date"])
    for col in ("Open", "High", "Low", "Close", "Volume"):
        shel.at[ts, col] = shel_anom[col]

    with caplog.at_level(logging.WARNING, logger="market_data.normalization"):
        norm_shel, raw_fp_shel, audit_shel = normalize_daily_bars_with_audit(
            shel, provider="yahoo", symbol="SHEL.L"
        )
    assert len(audit_shel) == 1
    assert audit_shel[0].date == "2024-05-01"
    assert audit_shel[0].field == "High"
    assert audit_shel[0].original == pytest.approx(2814.4806)
    assert audit_shel[0].repaired == pytest.approx(2871.919)
    assert audit_shel[0].reason == "clamped_high_below_open_close"
    assert norm_shel.at[ts, "High"] == pytest.approx(2871.919)
    assert raw_fp_shel != daily_frame_fingerprint(norm_shel)
    assert any(
        "market_data_repair" in rec.message and "SHEL.L" in rec.message for rec in caplog.records
    )

    # BP.L: 1 OHLC repair + 1 empty all-NaN zero-volume drop = 2 unique dates
    bp = _synthetic_frame(rows=fixtures["BP.L"]["rows"])
    bp_anom = fixtures["BP.L"]["anomalies"][0]
    bp_ts = pd.Timestamp(bp_anom["date"])
    for col in ("Open", "High", "Low", "Close", "Volume"):
        bp.at[bp_ts, col] = bp_anom[col]
    empty_ts = bp.index[50]
    for col, val in fixtures["BP.L"]["synthetic_empty_session"].items():
        bp.at[empty_ts, col] = np.nan if val is None else val

    norm_bp, _, audit_bp = normalize_daily_bars_with_audit(bp, provider="yahoo", symbol="BP.L")
    assert len(audit_bp) == 2
    assert {r.reason for r in audit_bp} == {
        "dropped_empty_nan_session",
        "clamped_high_below_open_close",
    }
    assert empty_ts not in norm_bp.index

    # RR.L: 3 bad bars where 2024-05-01 repairs BOTH High and Low (4 BarRepairRecords, 3 unique dates)
    rr = _synthetic_frame(rows=fixtures["RR.L"]["rows"])
    for anom in fixtures["RR.L"]["anomalies"]:
        a_ts = pd.Timestamp(anom["date"])
        for col in ("Open", "High", "Low", "Close", "Volume"):
            rr.at[a_ts, col] = anom[col]
    norm_rr, _, audit_rr = normalize_daily_bars_with_audit(rr, provider="yahoo", symbol="RR.L")
    assert len(audit_rr) == 4
    assert len({r.date for r in audit_rr}) == 3
    assert norm_rr.at[pd.Timestamp("2024-05-01"), "High"] == pytest.approx(412.9148)
    assert norm_rr.at[pd.Timestamp("2024-05-01"), "Low"] == pytest.approx(407.4159)


def test_nan_and_inf_fail_closed_rules() -> None:
    # Partial OHLC NaN fails closed even outside trailing 20 sessions
    partial = _synthetic_frame(rows=120)
    partial.at[partial.index[10], "Close"] = np.nan
    with pytest.raises(MarketDataProviderError):
        normalize_daily_bars_with_audit(partial, provider="yahoo", symbol="BP.L")

    # Volume > 0 with all-NaN OHLC fails closed
    vol_nonzero = _synthetic_frame(rows=120)
    for col in ("Open", "High", "Low", "Close"):
        vol_nonzero.at[vol_nonzero.index[10], col] = np.nan
    vol_nonzero.at[vol_nonzero.index[10], "Volume"] = 5000.0
    with pytest.raises(MarketDataProviderError):
        normalize_daily_bars_with_audit(vol_nonzero, provider="yahoo", symbol="BP.L")

    # +/- inf fails closed
    inf_frame = _synthetic_frame(rows=120)
    inf_frame.at[inf_frame.index[10], "Close"] = np.inf
    with pytest.raises(MarketDataProviderError):
        normalize_daily_bars_with_audit(inf_frame, provider="yahoo", symbol="BP.L")


def test_repair_thresholds_and_live_intraday_trimming() -> None:
    # <400 row IPO (N=120) allows 1 repair
    ipo = _synthetic_frame(rows=120)
    ipo.at[ipo.index[10], "High"] = ipo.at[ipo.index[10], "Open"] - 0.5
    _, _, ipo_audit = normalize_daily_bars_with_audit(ipo, provider="yahoo", symbol="IPO.L")
    assert len(ipo_audit) == 1

    # N=50 (<60) rejects any repair
    short = _synthetic_frame(rows=50)
    short.at[short.index[5], "High"] = short.at[short.index[5], "Open"] - 0.5
    with pytest.raises(MarketDataProviderError):
        normalize_daily_bars_with_audit(short, provider="yahoo", symbol="SHORT.L")

    # >5 unique dates on N=2000 fails closed
    too_many = _synthetic_frame(rows=2000)
    for i in range(6):
        too_many.at[too_many.index[10 + i], "High"] = (
            too_many.at[too_many.index[10 + i], "Open"] - 0.5
        )
    with pytest.raises(MarketDataProviderError):
        normalize_daily_bars_with_audit(too_many, provider="yahoo", symbol="MANY.L")

    # >5% drift fails closed (10x/100x corruption)
    drift_bad = _synthetic_frame(rows=120)
    drift_bad.at[drift_bad.index[10], "High"] = drift_bad.at[drift_bad.index[10], "Open"] * 0.5
    with pytest.raises(MarketDataProviderError):
        normalize_daily_bars_with_audit(drift_bad, provider="yahoo", symbol="DRIFT.L")

    # Non-positive price before clamp fails closed
    neg_bad = _synthetic_frame(rows=120)
    neg_bad.at[neg_bad.index[10], "Low"] = -1.0
    with pytest.raises(MarketDataProviderError):
        normalize_daily_bars_with_audit(neg_bad, provider="yahoo", symbol="NEG.L")

    # Trailing 20 sessions have 0 tolerance
    trailing_bad = _synthetic_frame(rows=120)
    trailing_bad.at[trailing_bad.index[-5], "High"] = (
        trailing_bad.at[trailing_bad.index[-5], "Open"] - 0.5
    )
    with pytest.raises(MarketDataProviderError):
        normalize_daily_bars_with_audit(trailing_bad, provider="yahoo", symbol="TRAIL.L")

    # In-progress live intraday bar dated AFTER required_session is trimmed BEFORE validation
    with_live_bar = _synthetic_frame(rows=121, end="2026-09-01")
    with_live_bar.at[pd.Timestamp("2026-09-01"), "High"] = np.nan
    trimmed, _, audit_trimmed = normalize_daily_bars_with_audit(
        with_live_bar,
        provider="yahoo",
        symbol="BP.L",
        max_session="2026-08-31",
    )
    assert trimmed.index[-1] == pd.Timestamp("2026-08-31")
    assert audit_trimmed == ()


def test_cache_schema_v3_round_trip_without_triple_normalization(monkeypatch, tmp_path) -> None:
    cache = MarketDataCache(tmp_path)
    frame = _synthetic_frame(rows=120, end="2026-08-31")
    audit = (
        BarRepairRecord(
            date="2026-04-01",
            field="High",
            original=499.0,
            repaired=500.0,
            reason="clamped_high_below_open_close",
        ),
    )
    saved = MarketDataResult(
        frame=frame,
        provider="yahoo",
        data_as_of="2026-08-31",
        feed="yahoo_daily",
        adjustment="auto_adjust_split_dividend",
        data_fingerprint="fp-post",
        raw_source_fingerprint="fp-raw",
        data_quality_audit=audit,
    )
    cache.save("SHEL.L", saved)

    # Spy: cache.load must NOT call normalize_daily_bars
    monkeypatch.setattr(
        "market_data.normalization.normalize_daily_bars",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache.load must not re-normalize")),
    )
    loaded = cache.load("yahoo", "SHEL.L", required_session="2026-08-31")
    assert loaded is not None
    assert loaded.cache_status == "hit"
    assert loaded.raw_source_fingerprint == "fp-raw"
    assert loaded.data_quality_audit == audit

    # Schema v2 entries are rejected
    path = cache._path("yahoo", "SHEL.L")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CACHE_SCHEMA_VERSION == 3
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert cache.load("yahoo", "SHEL.L", required_session="2026-08-31") is None


def test_yahoo_provider_never_passes_repair_true(monkeypatch) -> None:
    captured_kwargs: list[dict] = []

    def fake_download(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return _synthetic_frame(rows=80)

    monkeypatch.setattr("market_data.yahoo.yf.download", fake_download)
    YahooProvider().fetch_raw_daily_bars("SHEL.L", years=8)
    assert captured_kwargs
    assert "repair" not in captured_kwargs[0]


def test_provider_isolation_yahoo_outage_does_not_affect_alpaca_or_ready(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "market_data.service.latest_completed_trading_session",
        lambda **kwargs: pd.Timestamp("2026-08-31"),
    )
    us_frame = _synthetic_frame(rows=100, end="2026-08-31")
    alpaca = SimpleNamespace(
        name="alpaca",
        configured=True,
        fetch_daily_bars=lambda *_a, **_k: MarketDataResult(us_frame, "alpaca", "2026-08-31"),
    )
    yahoo = SimpleNamespace(
        name="yahoo",
        configured=True,
        fetch_daily_bars=lambda *_a, **_k: (_ for _ in ()).throw(
            MarketDataServiceError("yahoo outage")
        ),
    )
    service = MarketDataService([alpaca], uk_providers=[yahoo], cache=MarketDataCache(tmp_path))

    # Warm US success
    us_res = service.fetch_daily_bars("AAPL", years=8)
    assert us_res.provider == "alpaca"

    # Trip Yahoo circuit breaker with 3 consecutive failures
    for _ in range(3):
        with pytest.raises(MarketDataServiceError):
            service.fetch_daily_bars("SHEL.L", years=8)

    # US readiness and US fetches remain healthy; Yahoo is shown open in providers
    ready, details = service.readiness()
    assert ready is True
    assert details["last_success_session"] == "2026-08-31"
    assert details["providers"]["yahoo"]["circuit"] == "open"
    assert details["providers"]["alpaca"]["circuit"] == "closed"
    us_circuit_ok, _ = service.us_circuit_readiness()
    assert us_circuit_ok is True


def test_early_close_grace_window_and_collector_strictness(monkeypatch, tmp_path) -> None:
    # Black Friday 2026-11-27 closes at 13:00 ET = 18:00 UTC
    ctx_inside = completed_session_context(
        datetime(2026, 11, 27, 18, 10, tzinfo=UTC), exchange="NYSE"
    )
    assert ctx_inside.required_session == pd.Timestamp("2026-11-27")
    assert ctx_inside.previous_session == pd.Timestamp("2026-11-25")
    assert ctx_inside.in_post_close_grace is True

    ctx_outside = completed_session_context(
        datetime(2026, 11, 27, 18, 30, tzinfo=UTC), exchange="NYSE"
    )
    assert ctx_outside.in_post_close_grace is False

    # Public read uses grace fallback with Cache-Control: no-store and does NOT cache T-1 under T
    import data_pipeline
    from routes import market

    monkeypatch.setattr(
        "market_data.service.latest_completed_trading_session",
        lambda **kwargs: pd.Timestamp("2026-11-27"),
    )
    monkeypatch.setattr(
        "market_data.service.completed_session_context",
        lambda **kwargs: ctx_inside,
    )
    prev_frame = _synthetic_frame(rows=100, end="2026-11-25")
    provider = SimpleNamespace(
        name="alpaca",
        configured=True,
        fetch_daily_bars=lambda *_a, **_k: MarketDataResult(prev_frame, "alpaca", "2026-11-25"),
    )
    cache = MarketDataCache(tmp_path)
    service = MarketDataService([provider], cache=cache)
    monkeypatch.setattr(data_pipeline, "market_data_service", service)
    monkeypatch.setattr(market, "_fetch_history_intraday", lambda symbol: None)

    resp = CLIENT.get("/api/v1/history", params={"ticker": "AAPL"})
    assert resp.status_code == 200
    assert resp.headers.get("cache-control") == "no-store"
    assert resp.json()["post_close_grace_used"] is True
    assert resp.json()["as_of"] == "2026-11-25"
    assert cache.load("alpaca", "AAPL", required_session="2026-11-27") is None

    # Strictly exact-session for collector / score-ledger (allow_post_close_grace=False)
    with pytest.raises(MarketDataServiceError):
        service.fetch_daily_bars("AAPL", years=8, allow_post_close_grace=False)


def test_lse_usd_quote_unit_provenance_cpg_and_ihg(monkeypatch) -> None:
    from routes import market

    assert get_ticker_meta("CPG.L")["currency"] == "USD"
    assert get_ticker_meta("CPG.L")["currency_symbol"] == "$"
    assert get_ticker_meta("IHG.L")["currency"] == "USD"
    assert get_ticker_meta("IHG.L")["currency_symbol"] == "$"
    assert get_ticker_meta("SHEL.L")["currency"] == "GBp"
    assert get_ticker_meta("SHEL.L")["currency_symbol"] == "p"

    frame = _synthetic_frame(rows=60)
    monkeypatch.setattr(
        market,
        "_fetch_history_daily",
        lambda symbol: MarketDataResult(frame, "yahoo", frame.index[-1].date().isoformat()),
    )
    monkeypatch.setattr(market, "_fetch_history_intraday", lambda symbol: None)

    cpg = CLIENT.get("/api/v1/history", params={"ticker": "CPG.L"}).json()
    assert cpg["currency"] == "USD"
    assert cpg["currency_symbol"] == "$"

    shel = CLIENT.get("/api/v1/history", params={"ticker": "SHEL.L"}).json()
    assert shel["currency"] == "GBp"
    assert shel["currency_symbol"] == "p"
