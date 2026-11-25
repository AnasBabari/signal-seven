"""P1 Conformal Price Band Calibration tests: causality, regime scaling, latency, and artifact parity."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from services import simple_forecast as sf


def _ohlcv_frame(
    rows: int = 1200, *, seed: int = 123, tail_vol_multiplier: float = 1.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.012, size=rows)
    if tail_vol_multiplier != 1.0:
        returns[-40:] *= float(tail_vol_multiplier)
    close = 120.0 * np.exp(np.cumsum(returns))
    open_ = close * np.exp(rng.normal(0.0, 0.002, size=rows))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.004, size=rows)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.004, size=rows)))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(500_000, 3_000_000, size=rows).astype(float),
        },
        index=pd.bdate_range("2020-01-02", periods=rows),
    )


def test_conformal_band_emits_pass_metadata_and_round_trips_disk_artifact(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(sf.settings, "forecast_model_cache_dir", str(tmp_path))
    sf.clear_forecast_cache()
    frame = _ohlcv_frame(1200, seed=11)

    t0 = time.perf_counter()
    first = sf.train_and_forecast("MSFT", frame, model_name="ridge")
    cold_ms = (time.perf_counter() - t0) * 1000.0

    cal = first["interval_calibration"]
    assert cal["method"] == "volatility_scaled_online_split_conformal"
    assert cal["vol_scale"] == "blend_ewma94_v60"
    assert cal["m5_gate_verdict"] == "PASS"
    assert cal["nominal_pointwise_coverage"] == 0.80
    assert cal["matured_calibration_samples"] >= sf.CONFORMAL_MIN_MATURED
    assert first["backtest"]["interval_coverage"] == "validated_80pct_pointwise_conformal"
    # Cold conformal Ridge must remain fast (<500ms even on slow CI runners)
    assert cold_ms < 1500.0

    # Disk artifact reload must reproduce exact bounds and calibration metadata
    with sf._cache_lock:
        sf._cache.clear()
    reloaded = sf.train_and_forecast("MSFT", frame, model_name="ridge")
    assert reloaded["lower_prices"] == first["lower_prices"]
    assert reloaded["upper_prices"] == first["upper_prices"]
    assert reloaded["interval_calibration"] == cal


def test_conformal_band_is_strictly_causal_zero_future_leakage() -> None:
    frame = _ohlcv_frame(900, seed=29)
    cutoff = 750
    sub_a = frame.iloc[:cutoff].copy()
    ds_a = sf.build_dataset(sub_a)
    res_a = sf._compute_conformal_band_offsets(sub_a, ds_a)
    assert res_a is not None

    # Mutate bars strictly after cutoff and re-slice to cutoff: identical bit-for-bit
    mutated = frame.copy()
    mutated.iloc[cutoff:] *= 3.5
    sub_b = mutated.iloc[:cutoff].copy()
    ds_b = sf.build_dataset(sub_b)
    res_b = sf._compute_conformal_band_offsets(sub_b, ds_b)
    assert res_b is not None

    np.testing.assert_allclose(res_a[0], res_b[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(res_a[1], res_b[1], rtol=0.0, atol=0.0)


def test_conformal_band_widens_proportionally_in_elevated_volatility_regime() -> None:
    calm_frame = _ohlcv_frame(1000, seed=47, tail_vol_multiplier=1.0)
    elev_frame = _ohlcv_frame(1000, seed=47, tail_vol_multiplier=2.5)

    calm_res = sf._compute_conformal_band_offsets(calm_frame, sf.build_dataset(calm_frame))
    elev_res = sf._compute_conformal_band_offsets(elev_frame, sf.build_dataset(elev_frame))
    assert calm_res is not None and elev_res is not None

    calm_width = calm_res[1] - calm_res[0]
    elev_width = elev_res[1] - elev_res[0]
    assert np.all(elev_width > calm_width * 1.4)
    assert elev_res[2]["vol_scale_latest"] > calm_res[2]["vol_scale_latest"] * 1.4


def test_random_forest_explicit_request_uses_unverified_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sf.settings, "forecast_model_cache_dir", str(tmp_path))
    sf.clear_forecast_cache()
    frame = _ohlcv_frame(650, seed=53)
    res = sf.train_and_forecast("AAPL", frame, model_name="random_forest")
    assert res["interval_calibration"]["method"] == "static_validation_quantiles"
    assert res["interval_calibration"]["m5_gate_verdict"] == "UNVERIFIED_FALLBACK"
    assert res["backtest"]["interval_coverage"] == "unavailable"
