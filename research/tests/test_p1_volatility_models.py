"""P1 Volatility ML unit tests: causal Duan-smeared HAR-RV, v5_duan_har policy, and G3 v2 scale invariance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from services import volatility_snapshot
from services.volatility_contract import (
    DEFAULT_VOLATILITY_POLICY,
    VOLATILITY_POLICY_MAPS,
    resolve_auto_model_policy,
)

from research.volatility_structure.g3_v2 import (
    G3_V2_FEATURES,
    build_g3_v2_features,
    panel_rolling_base_60d,
)


def _synthetic_ohlcv(n: int = 420, seed: int = 101, sigma: float = 0.015) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, sigma, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = close * np.exp(rng.normal(0.0, sigma * 0.2, size=n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, sigma * 0.4, size=n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, sigma * 0.4, size=n)))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=pd.bdate_range("2022-01-03", periods=n),
    )


def test_duan_smeared_har_eliminates_geometric_mean_downward_bias() -> None:
    df = _synthetic_ohlcv(n=600, seed=7, sigma=0.018)
    logret = np.diff(np.log(df["Close"].to_numpy()), prepend=np.log(df["Close"].iloc[0]))
    rv = logret**2
    rv[0] = 0.0

    unsmeared = volatility_snapshot.causal_log_har_forecasts(rv, (5, 10, 20), smearing=False)
    smeared = volatility_snapshot.causal_log_har_forecasts(rv, (5, 10, 20), smearing=True)
    latest = volatility_snapshot.causal_log_har_forecasts(
        rv, (5, 10, 20), latest_only=True, smearing=True
    )

    # Bit-exact parity between full-series and latest_only Duan smearing
    np.testing.assert_allclose(latest[-1], smeared[-1], rtol=1e-12, atol=1e-15)

    # True expected 5/10/20d cumulative variance under sigma=0.018 is H * 0.018^2
    true_var = np.array([5, 10, 20], dtype=float) * (0.018**2)
    valid = np.isfinite(smeared).all(axis=1)
    unsmeared_ratio = unsmeared[valid].mean(axis=0) / true_var
    smeared_ratio = smeared[valid].mean(axis=0) / true_var

    # Legacy unsmeared severely underforecasts (<0.45x); Duan-smeared centers near 1.0x
    assert np.all(unsmeared_ratio < 0.45)
    assert np.all((smeared_ratio > 0.75) & (smeared_ratio < 1.25))


def test_duan_smeared_har_is_strictly_causal_zero_future_leakage() -> None:
    df = _synthetic_ohlcv(n=300, seed=19)
    logret = np.diff(np.log(df["Close"].to_numpy()), prepend=np.log(df["Close"].iloc[0]))
    rv = logret**2
    rv[0] = 0.0

    cutoff = 200
    base = volatility_snapshot.causal_log_har_forecasts(rv, (1, 5, 10, 20), smearing=True)
    mutated = rv.copy()
    mutated[cutoff + 1 :] *= 25.0  # Extreme shock strictly after cutoff
    after = volatility_snapshot.causal_log_har_forecasts(mutated, (1, 5, 10, 20), smearing=True)

    np.testing.assert_allclose(
        base[: cutoff + 1], after[: cutoff + 1], rtol=0.0, atol=0.0, equal_nan=True
    )


def test_v5_duan_har_policy_is_default_and_v3_rollback_preserved() -> None:
    assert DEFAULT_VOLATILITY_POLICY == "v5_duan_har"
    assert resolve_auto_model_policy("v5_duan_har") == {
        1: "garch_11",
        5: "har_rv",
        10: "har_rv",
        20: "har_rv",
    }
    assert resolve_auto_model_policy("v3_baseline") == {
        1: "garch_11",
        5: "rolling_mean",
        10: "rolling_mean",
        20: "rolling_mean",
    }
    assert set(VOLATILITY_POLICY_MAPS) >= {"v3_baseline", "v4_g3_h5", "v4_g3", "v5_duan_har"}


def test_g3_v2_features_are_scale_invariant_and_causal() -> None:
    df = _synthetic_ohlcv(n=350, seed=31, sigma=0.02)
    f1 = build_g3_v2_features(df)
    assert list(f1.columns) == list(G3_V2_FEATURES)
    assert len(G3_V2_FEATURES) == 24

    # Exact 0.5x volatility scaling across OHLC log-returns
    scaled = df.copy()
    c0 = df["Close"].iloc[0]
    log_c = np.log(df["Close"].to_numpy())
    scaled["Close"] = np.exp(np.log(c0) + 0.5 * (log_c - np.log(c0)))
    scaled["Open"] = scaled["Close"] * np.exp(0.5 * np.log(df["Open"] / df["Close"]))
    scaled["High"] = scaled["Close"] * np.exp(0.5 * np.log(df["High"] / df["Close"]))
    scaled["Low"] = scaled["Close"] * np.exp(0.5 * np.log(df["Low"] / df["Close"]))
    f2 = build_g3_v2_features(scaled)

    valid = f1.dropna().index
    max_diff = float(np.max(np.abs(f1.loc[valid].to_numpy() - f2.loc[valid].to_numpy())))
    assert max_diff < 1e-6

    # Strict causality: mutating future bars produces zero change before cutoff
    mutated = df.copy()
    mutated.iloc[250:] *= 1.8
    f_mut = build_g3_v2_features(mutated)
    np.testing.assert_allclose(
        f1.iloc[:250].to_numpy(),
        f_mut.iloc[:250].to_numpy(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )

    # 60d rolling base scales quadratically with return scale (0.5^2 = 0.25x)
    b1 = panel_rolling_base_60d(df, 5)
    b2 = panel_rolling_base_60d(scaled, 5)
    assert b2 == pytest.approx(0.25 * b1, rel=1e-10)
