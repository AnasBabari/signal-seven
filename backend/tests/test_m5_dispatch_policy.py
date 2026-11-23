"""Milestone 05 Part 2 tests: serving & collector dispatch architecture and provenance binding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app
from config import Settings, settings
from market_data.normalization import daily_frame_fingerprint
from routes import volatility
from scripts.collect_live_forecasts import ApiResult, run_live_collection, run_preflight
from services import g3_volatility, volatility_snapshot
from services.forecast_ledger import ForecastLedger
from services.live_collection import LIVE_MODEL_POLICY_V1
from services.live_volatility import build_live_volatility_forecast
from services.volatility_contract import (
    DEFAULT_VOLATILITY_POLICY,
    VOLATILITY_POLICY_MAPS,
    resolve_auto_model_policy,
    resolve_volatility_policy_name,
)

CLIENT = TestClient(app)


def _synthetic_ohlcv(rows: int = 520, *, seed: int = 23, end: str = "2026-09-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.012, size=rows)
    close = 150.0 * np.exp(np.cumsum(returns))
    open_ = close * np.exp(rng.normal(0.0, 0.002, size=rows))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.012, size=rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.012, size=rows))
    frame = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(500_000, 4_000_000, size=rows).astype(float),
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


def test_volatility_default_policy_maps_and_h1_garch_invariant(monkeypatch) -> None:
    assert DEFAULT_VOLATILITY_POLICY == "v3_baseline"
    assert settings.volatility_default_policy == "v3_baseline"
    assert resolve_volatility_policy_name() == "v3_baseline"

    for name, expected in (
        (
            "v3_baseline",
            {1: "garch_11", 5: "rolling_mean", 10: "rolling_mean", 20: "rolling_mean"},
        ),
        (
            "v4_g3_h5",
            {1: "garch_11", 5: "gpu_g3", 10: "rolling_mean", 20: "rolling_mean"},
        ),
        (
            "v4_g3",
            {1: "garch_11", 5: "gpu_g3", 10: "gpu_g3", 20: "gpu_g3"},
        ),
    ):
        resolved = resolve_auto_model_policy(name)
        assert resolved == expected
        # Every policy must keep H=1 on garch_11 because g3_h1.onnx does not exist.
        assert resolved[1] == "garch_11"
        assert VOLATILITY_POLICY_MAPS[name][1] == "garch_11"

    # Settings validator normalizes case/whitespace and rejects unknown policies.
    cfg = Settings(_env_file=None, volatility_default_policy="  V4_G3_H5 ")
    assert cfg.volatility_default_policy == "v4_g3_h5"
    with pytest.raises(ValueError):
        Settings(_env_file=None, volatility_default_policy="invalid_policy")
    with pytest.raises(ValueError, match="VOLATILITY_DEFAULT_POLICY"):
        resolve_auto_model_policy("invalid_policy")

    # Both settings and env overrides are respected by resolve_auto_model_policy().
    monkeypatch.setenv("VOLATILITY_DEFAULT_POLICY", "v4_g3")
    assert resolve_auto_model_policy()[5] == "gpu_g3"
    monkeypatch.setattr(settings, "volatility_default_policy", "v4_g3_h5")
    assert resolve_auto_model_policy()[5] == "gpu_g3"
    assert resolve_auto_model_policy()[10] == "rolling_mean"


def test_auto_dispatches_to_gpu_g3_on_single_and_batch_routes(monkeypatch) -> None:
    volatility_snapshot.clear_snapshot_cache()
    frame = _synthetic_ohlcv(520)
    monkeypatch.setattr(
        volatility_snapshot,
        "_download_ohlcv",
        lambda _ticker, **_kwargs: frame.copy(),
    )

    # 1. Under v4_g3_h5: H=1 -> garch_11, H=5 -> gpu_g3, H=10/20 -> rolling_mean
    monkeypatch.setattr(settings, "volatility_default_policy", "v4_g3_h5")
    snapshot = volatility_snapshot.build_volatility_inference_snapshot("MSFT")

    h1 = build_live_volatility_forecast(snapshot, horizon=1, model="auto")
    h5 = build_live_volatility_forecast(snapshot, horizon=5, model="auto")
    h10 = build_live_volatility_forecast(snapshot, horizon=10, model="auto")
    assert h1["forecast"]["model"] == "garch_11"
    assert h5["forecast"]["model"] == "gpu_g3"
    assert h5["forecast"]["requested_model"] == "auto"
    assert h5["evidence"]["model_status"] == "learned_model"
    assert h5["evidence"]["default_policy"] == "v4_g3_h5"
    assert h10["forecast"]["model"] == "rolling_mean"

    # 2. Under v4_g3: batch route with model=auto dispatches 5/10/20 to gpu_g3 in ONE feature pass
    monkeypatch.setattr(settings, "volatility_default_policy", "v4_g3")
    g3_calls = {"count": 0}
    real_build_g3 = g3_volatility.build_g3_features

    def spy_build_g3(df):
        g3_calls["count"] += 1
        return real_build_g3(df)

    monkeypatch.setattr(g3_volatility, "build_g3_features", spy_build_g3)

    batch_resp = CLIENT.get(
        "/api/v1/volatility/forecast/batch",
        params={"ticker": "MSFT", "horizons": "1,5,10,20", "model": "auto"},
    )
    assert batch_resp.status_code == 200
    assert g3_calls["count"] == 1
    body = batch_resp.json()
    assert body["forecasts"]["1"]["forecast"]["model"] == "garch_11"
    for horizon in ("5", "10", "20"):
        fc = body["forecasts"][horizon]
        assert fc["forecast"]["model"] == "gpu_g3"
        assert fc["forecast"]["requested_model"] == "auto"
        assert fc["evidence"]["model_status"] == "learned_model"
        assert fc["evidence"]["default_policy"] == "v4_g3"
        assert fc["evidence"]["auto_model_policy"] == {
            "1": "garch_11",
            "5": "gpu_g3",
            "10": "gpu_g3",
            "20": "gpu_g3",
        }
    volatility_snapshot.clear_snapshot_cache()


def test_v1_collector_remains_pinned_when_public_default_policy_is_v4_g3(
    monkeypatch, tmp_path
) -> None:
    """Changing public VOLATILITY_DEFAULT_POLICY must never break V1 collector continuity."""
    from routes.common import limiter

    monkeypatch.setattr(limiter, "enabled", False)
    volatility_snapshot.clear_snapshot_cache()
    frame = _synthetic_ohlcv(520, end="2026-09-02")
    monkeypatch.setattr(
        volatility_snapshot,
        "_download_ohlcv",
        lambda _ticker, **_kwargs: frame.copy(),
    )
    ledger = ForecastLedger(tmp_path / "v1_pinned.db")
    monkeypatch.setattr(volatility, "get_forecast_ledger", lambda: ledger)
    monkeypatch.setenv("FORECAST_COLLECTOR_TOKEN", "collector-secret-token")

    # Switch public default policy to v4_g3 (so public model=auto serves gpu_g3 for H=5).
    monkeypatch.setattr(settings, "volatility_default_policy", "v4_g3")

    public_auto = CLIENT.get(
        "/api/v1/volatility/forecast",
        params={"ticker": "MSFT", "horizon": 5, "model": "auto"},
    )
    assert public_auto.status_code == 200
    assert public_auto.json()["forecast"]["model"] == "gpu_g3"

    # V1 collector POST must stay pinned to LIVE_MODEL_POLICY_V1[5] == "rolling_mean".
    collect_resp = CLIENT.post(
        "/api/v1/volatility/collect",
        params={"ticker": "MSFT", "horizon": 5},
        headers={"Authorization": "Bearer collector-secret-token"},
    )
    assert collect_resp.status_code == 200
    collected = collect_resp.json()
    assert collected["forecast"]["model"] == LIVE_MODEL_POLICY_V1[5] == "rolling_mean"
    assert collected["evidence"]["ledger_write"] == "recorded_live"

    # Collector script preflight requests explicit LIVE_MODEL_POLICY_V1[horizon] so preflight
    # fingerprint matches POST /collect even under public v4_g3.
    requested_models: list[str] = []

    class _AsgiCollectorClient:
        def request(self, method, path, *, params=None, token=None, attempts=2):
            del attempts
            headers = {"Authorization": f"Bearer {token}"} if token else {}
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
            if method == "GET" and params and "model" in params:
                requested_models.append(str(params["model"]))
            resp = CLIENT.request(method, path, params=params, headers=headers)
            return ApiResult(resp.status_code, resp.json())

    asgi_client = _AsgiCollectorClient()
    preflight = run_preflight(
        asgi_client,  # type: ignore[arg-type]
        expected_session="2026-09-02",
        run_timestamp="2026-09-02T22:30:00+00:00",
        interval_seconds=0,
    )
    assert preflight["batch_status"] == "dry_run_passed"
    assert "auto" not in requested_models
    assert set(requested_models) == {"garch_11", "rolling_mean"}

    live_manifest = run_live_collection(
        asgi_client,  # type: ignore[arg-type]
        preflight,
        token="collector-secret-token",
        interval_seconds=0,
    )
    assert live_manifest["batch_status"] == "complete"
    assert live_manifest["succeeded_count"] == 60
    assert live_manifest["failed_count"] == 0
    volatility_snapshot.clear_snapshot_cache()


def test_m5_artifact_sha256_provenance_binding_and_invalidation(monkeypatch) -> None:
    volatility_snapshot.clear_snapshot_cache()
    frame = _synthetic_ohlcv(520)
    monkeypatch.setattr(
        volatility_snapshot,
        "_download_ohlcv",
        lambda _ticker, **_kwargs: frame.copy(),
    )
    snapshot = volatility_snapshot.build_volatility_inference_snapshot("MSFT")

    expected_shas = {
        5: "8bccde12efb4bf5089ce3cd704cc027653f916f0547ddb5a4404587513d76a93",
        10: "5014ff1bf2c31e19e00061099748bdba75ca8f4451b0602d7f3384a044f89fbc",
        20: "79ee8efcf10322ed6fd70f6a382a9aec430bce4eef7e410273b1a30d54665325",
    }
    for horizon, expected_sha in expected_shas.items():
        res = build_live_volatility_forecast(snapshot, horizon=horizon, model="gpu_g3")
        prov = res["evidence"]["m5_validation_provenance"]
        assert prov is not None
        assert prov["artifact_sha256"] == expected_sha
        assert prov["horizon"] == horizon
        assert prov["comparator"] == "rolling_mean_60d"
        assert prov["evaluation_snapshot"] == "m5_purged_oos_2023-11-16_2026-11-20"
        assert prov["promote_g3_to_auto"] is False
        # Because overall M5 promotion gate returned FAIL, test_evidence_qlike_vs_rolling is None.
        assert res["evidence"]["test_evidence_qlike_vs_rolling"] is None

    # A different or unverified artifact SHA-256 fails closed to None provenance.
    assert g3_volatility.get_m5_validation_provenance(5, "0" * 64) is None
    assert g3_volatility.get_m5_validation_provenance(10, expected_shas[5]) is None
    volatility_snapshot.clear_snapshot_cache()
