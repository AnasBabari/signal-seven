"""Unit tests for Milestone 05 empirical validation statistics, zero-leakage & gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from services.simple_forecast import FORECAST_DAYS, build_features, train_and_forecast

from research.validation.validate_m5_models import (
    bartlett_hac_summary,
    circular_block_bootstrap,
    circular_block_bootstrap_ratio,
    evaluate_price_band_gate,
    evaluate_volatility_horizon_gate,
    exact_served_ridge_at_origin,
    holm_bonferroni,
    mincer_zarnowitz_panel_test,
    strict_matured_conformal_quantiles,
    verify_locked_g3_artifacts,
)
from research.volatility_structure.evaluation import hac_mean


def _synthetic_ohlcv(n_rows: int = 680, seed: int = 1234) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_rows)
    rets = rng.normal(0.0003, 0.015, size=n_rows)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = close * np.exp(rng.normal(0.0, 0.004, size=n_rows))
    spread = np.abs(rng.normal(0.006, 0.003, size=n_rows))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.integers(500_000, 5_000_000, size=n_rows).astype(float)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return df


def test_verify_locked_g3_artifacts_checksums_and_h1_absent() -> None:
    report = verify_locked_g3_artifacts()
    assert report["g3_h1_onnx_exists"] is False
    assert report["h1_policy_preserved"] == "garch_11"
    for h in ("5", "10", "20"):
        info = report["horizons"][h]
        assert info["checksum_verified"] is True
        assert info["features_match_serving_code"] is True
        assert info["graph_base_score"] == pytest.approx(0.5)
        assert info["n_features"] == 22
        assert len(info["onnx_sha256"]) == 64


def test_bartlett_hac_matches_research_and_widens_for_autocorrelation() -> None:
    rng = np.random.default_rng(42)
    eps = rng.normal(0.0, 1.0, size=400)
    # Positively autocorrelated MA(4) series simulating H=5 overlapping forecast errors
    ma4 = np.convolve(eps, np.ones(5) / 5.0, mode="valid") + 0.15

    res = bartlett_hac_summary(ma4, bandwidth=4)
    ref = hac_mean(ma4, 4)
    assert res["mean"] == pytest.approx(ref["mean_loss_improvement"], rel=1e-12)
    assert res["se"] == pytest.approx(ref["se"], rel=1e-12)
    assert res["p_two_sided"] == pytest.approx(ref["p_two_sided"], rel=1e-12)

    naive_se = bartlett_hac_summary(ma4, bandwidth=0)["se"]
    assert res["se"] > naive_se * 1.4


def test_circular_block_bootstrap_determinism_and_ratio() -> None:
    rng = np.random.default_rng(99)
    series = rng.normal(0.05, 0.2, size=300)
    b1 = circular_block_bootstrap(series, block_length=10, n_boot=500, seed=77)
    b2 = circular_block_bootstrap(series, block_length=10, n_boot=500, seed=77)
    assert b1 == b2
    assert b1["ci95"][0] < b1["mean"] < b1["ci95"][1]

    den = np.full(200, 0.03) + rng.uniform(0.0, 0.005, size=200)
    num_better = den * 0.90
    num_worse = den * 1.08
    r_better = circular_block_bootstrap_ratio(num_better, den, block_length=10, n_boot=400, seed=7)
    r_worse = circular_block_bootstrap_ratio(num_worse, den, block_length=10, n_boot=400, seed=7)
    assert r_better["ci95"][1] < 1.0
    assert r_better["p_one_sided_lt_one"] < 0.01
    assert r_worse["ci95"][0] > 1.0
    assert r_worse["p_one_sided_lt_one"] > 0.95


def test_holm_bonferroni_exact_step_down() -> None:
    raw = [0.01, 0.04, 0.03, 0.005]
    adj = holm_bonferroni(raw)
    # Sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    expected = [0.03, 0.06, 0.06, 0.02]
    assert adj == pytest.approx(expected, rel=1e-12)
    assert holm_bonferroni([0.8, 0.9]) == pytest.approx([1.0, 1.0])


def test_mincer_zarnowitz_passes_calibrated_and_catches_level_and_slope_bias() -> None:
    rng = np.random.default_rng(2026)
    n_dates = 400
    n_assets = 4
    dates = np.repeat(pd.bdate_range("2023-01-02", periods=n_dates), n_assets)
    f_vals = rng.uniform(0.04, 0.16, size=n_dates * n_assets)
    noise = rng.normal(0.0, 0.015, size=n_dates * n_assets)

    # 1. Well-calibrated: y = 0 + 1.0 * f + noise
    y_cal = np.maximum(f_vals + noise, 1e-6)
    res_cal = mincer_zarnowitz_panel_test(y_cal, f_vals, dates, bandwidth=4)
    assert res_cal["mz_gate_pass"] is True
    assert abs(res_cal["intercept_rel"]) < 0.05
    assert 0.95 <= res_cal["slope"] <= 1.05

    # 2. Pure level bias at slope == 1.0: y = 0.05 + 1.0 * f + noise (alpha_rel ~= +0.33)
    y_level_bias = 0.05 + f_vals + noise
    res_level = mincer_zarnowitz_panel_test(y_level_bias, f_vals, dates, bandwidth=4)
    assert 0.95 <= res_level["slope"] <= 1.05  # slope looks perfect!
    assert res_level["p_wald_chi2"] < 1e-6
    assert res_level["checks"]["intercept_point_in_bounds"] is False
    assert res_level["checks"]["mean_ratio_in_bounds"] is False
    assert res_level["mz_gate_pass"] is False

    # 3. Slope distortion with unbiased mean: y = 0.05 + 0.5 * f + noise (mean_y ~= mean_f = 0.10)
    y_slope_bias = 0.05 + 0.5 * f_vals + noise
    res_slope = mincer_zarnowitz_panel_test(y_slope_bias, f_vals, dates, bandwidth=4)
    assert 0.95 <= res_slope["mean_forecast_to_realized_ratio"] <= 1.05  # mean matches!
    assert res_slope["checks"]["slope_point_in_bounds"] is False
    assert res_slope["mz_gate_pass"] is False


def test_mincer_zarnowitz_driscoll_kraay_invariance_to_cross_sectional_cloning() -> None:
    rng = np.random.default_rng(7)
    n_dates = 250
    dates_1 = pd.bdate_range("2024-01-02", periods=n_dates)
    f_1 = rng.uniform(0.05, 0.15, size=n_dates)
    y_1 = f_1 + rng.normal(0.0, 0.02, size=n_dates)

    # Clone identical correlated series 10x across dates: Driscoll-Kraay SE must NOT shrink by sqrt(10)
    res_1 = mincer_zarnowitz_panel_test(y_1, f_1, dates_1, bandwidth=4)
    res_10 = mincer_zarnowitz_panel_test(
        np.tile(y_1, 10), np.tile(f_1, 10), np.tile(dates_1, 10), bandwidth=4
    )
    assert res_10["slope_se"] == pytest.approx(res_1["slope_se"], rel=1e-10)
    assert res_10["intercept_se"] == pytest.approx(res_1["intercept_se"], rel=1e-10)
    assert res_10["wald_chi2"] == pytest.approx(res_1["wald_chi2"], rel=1e-10)


def test_volatility_horizon_gate_strict_clauses() -> None:
    passing_comp = {
        "mean_delta_qlike": 0.10,
        "asset_win_rate": 0.80,
        "hac_primary": {"p_holm": 0.001, "ci95": [0.05, 0.15]},
        "hac_2x": {"ci95": [0.04, 0.16]},
        "block_bootstrap": {"ci95": [0.04, 0.16]},
    }
    passing_mz = {"mz_gate_pass": True}

    ok = evaluate_volatility_horizon_gate(
        vs_rolling=passing_comp,
        vs_har=passing_comp,
        us_vs_rolling=passing_comp,
        lse_vs_rolling=passing_comp,
        mz_g3=passing_mz,
    )
    assert ok["verdict"] == "PASS"
    assert ok["failed_clauses"] == []

    # Fails if LSE win rate drops below 70%
    weak_lse = {**passing_comp, "asset_win_rate": 0.60}
    fail_lse = evaluate_volatility_horizon_gate(
        vs_rolling=passing_comp,
        vs_har=passing_comp,
        us_vs_rolling=passing_comp,
        lse_vs_rolling=weak_lse,
        mz_g3=passing_mz,
    )
    assert fail_lse["verdict"] == "FAIL"
    assert "win_rate_ge_70pct_vs_rolling_pooled_us_lse" in fail_lse["failed_clauses"]

    # Fails if MZ calibration fails
    fail_mz = evaluate_volatility_horizon_gate(
        vs_rolling=passing_comp,
        vs_har=passing_comp,
        us_vs_rolling=passing_comp,
        lse_vs_rolling=passing_comp,
        mz_g3={"mz_gate_pass": False},
    )
    assert fail_mz["verdict"] == "FAIL"
    assert "mincer_zarnowitz_joint_calibration" in fail_mz["failed_clauses"]


def test_strict_matured_conformal_quantiles_enforces_exact_target_maturity() -> None:
    positions = np.arange(0, 101, dtype=np.int64)
    residuals = np.zeros(101, dtype=np.float64)
    # Poison origin s=100 for horizon H=7 (matures at 107, NOT 106)
    residuals[100] = 999.0

    # At origin 106, s=100 has NOT matured (100 + 7 = 107 > 106) -> poison MUST be excluded
    q_before = strict_matured_conformal_quantiles(
        residuals, positions, current_origin_position=106, horizon_days=7, min_matured=50
    )
    assert q_before is not None
    assert q_before[0] == 0.0
    assert q_before[1] == 0.0
    assert q_before[2] == 100  # positions 0..99 only

    # At origin 107, s=100 HAS matured (100 + 7 <= 107) -> poison IS now admitted
    q_after = strict_matured_conformal_quantiles(
        residuals,
        positions,
        current_origin_position=107,
        horizon_days=7,
        alpha=0.01,
        min_matured=50,
    )
    assert q_after is not None
    assert q_after[2] == 101
    assert q_after[1] == 999.0


def test_exact_served_ridge_zero_future_leakage_and_serving_parity() -> None:
    df = _synthetic_ohlcv(n_rows=660, seed=55)
    origin_pos = 620

    # Build mutated frame with extreme future shock strictly AFTER origin_pos
    df_mutated = df.copy()
    df_mutated.iloc[origin_pos + 1 :, df_mutated.columns.get_loc("Close")] *= 10.0
    df_mutated.iloc[origin_pos + 1 :, df_mutated.columns.get_loc("Open")] *= 10.0
    df_mutated.iloc[origin_pos + 1 :, df_mutated.columns.get_loc("High")] *= 10.0
    df_mutated.iloc[origin_pos + 1 :, df_mutated.columns.get_loc("Low")] *= 10.0

    def _run_origin(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        feats = build_features(frame)
        log_c = np.log(frame["Close"].to_numpy(dtype=np.float64))
        pos_map = pd.Series(np.arange(len(frame), dtype=np.int64), index=frame.index)
        fpos = pos_map.loc[feats.index].to_numpy(dtype=np.int64)
        targets = np.full((len(feats), FORECAST_DAYS), np.nan, dtype=np.float64)
        for i, p in enumerate(fpos):
            if p + FORECAST_DAYS < len(frame):
                for h in range(FORECAST_DAYS):
                    targets[i, h] = float(log_c[p + h + 1] - log_c[p])
        return exact_served_ridge_at_origin(feats, targets, fpos, origin_pos)

    out_clean = _run_origin(df)
    out_mut = _run_origin(df_mutated)

    # Zero leakage: mutating bars after origin_pos produces bit-for-bit identical forecasts & bands
    np.testing.assert_array_equal(out_clean["forecast_returns"], out_mut["forecast_returns"])
    np.testing.assert_array_equal(out_clean["lower_returns"], out_mut["lower_returns"])
    np.testing.assert_array_equal(out_clean["upper_returns"], out_mut["upper_returns"])

    # Exact parity against production simple_forecast.train_and_forecast(..., model_name="ridge")
    sliced = df.iloc[: origin_pos + 1].copy()
    served = train_and_forecast("MSFT", sliced, model_name="ridge")
    cur_price = float(sliced["Close"].iloc[-1])
    np.testing.assert_allclose(
        cur_price * np.exp(out_clean["forecast_returns"]),
        np.asarray(served["predicted_prices"]),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cur_price * np.exp(out_clean["lower_returns"]),
        np.asarray(served["lower_prices"]),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cur_price * np.exp(out_clean["upper_returns"]),
        np.asarray(served["upper_prices"]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_price_band_gate_rejects_regime_collapse_and_tail_asymmetry() -> None:
    good_h = {h: 0.80 for h in range(1, 8)}
    ok = evaluate_price_band_gate(
        pooled_pointwise_coverage=0.80,
        per_horizon_coverages=good_h,
        us_pointwise_coverage=0.81,
        lse_pointwise_coverage=0.79,
        calm_pointwise_coverage=0.83,
        elevated_pointwise_coverage=0.76,
        pooled_lower_tail_miss=0.10,
        pooled_upper_tail_miss=0.10,
    )
    assert ok["verdict"] == "PASS"

    # Elevated volatility collapse (< 75%) fails even when pooled is 80%
    fail_regime = evaluate_price_band_gate(
        pooled_pointwise_coverage=0.80,
        per_horizon_coverages=good_h,
        us_pointwise_coverage=0.80,
        lse_pointwise_coverage=0.80,
        calm_pointwise_coverage=0.84,
        elevated_pointwise_coverage=0.68,
        pooled_lower_tail_miss=0.10,
        pooled_upper_tail_miss=0.10,
    )
    assert fail_regime["verdict"] == "FAIL"
    assert "calm_and_elevated_regimes_in_75_85" in fail_regime["failed_clauses"]

    # Asymmetric tail miss (18% lower, 2% upper) fails even when pooled is 80%
    fail_tails = evaluate_price_band_gate(
        pooled_pointwise_coverage=0.80,
        per_horizon_coverages=good_h,
        us_pointwise_coverage=0.80,
        lse_pointwise_coverage=0.80,
        calm_pointwise_coverage=0.80,
        elevated_pointwise_coverage=0.80,
        pooled_lower_tail_miss=0.18,
        pooled_upper_tail_miss=0.02,
    )
    assert fail_tails["verdict"] == "FAIL"
    assert "tail_symmetry_5_to_15_each_tail" in fail_tails["failed_clauses"]
