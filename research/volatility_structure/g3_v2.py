"""Scale-free, 60d-anchored G3 v2 volatility feature builder (research module).

Every one of the 24 features is strictly dimensionless and invariant to constant
volatility scaling ``r_t -> c * r_t`` (verified to <1e-6 max absolute error),
eliminating G3 v1's cross-market US-to-LSE variance level bias. Base anchor is
the production 60-day close-to-close variance ``H * Var_60d(ddof=1)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from services.g3_volatility import _har_log_components, _range_variances, _yang_zhang_var

G3_V2_FEATURE_SCHEMA = "vol-scale-free-v2"
EPS_VAR = 1e-10

G3_V2_FEATURES: tuple[str, ...] = (
    "log_rv5_60",
    "log_rv10_60",
    "log_rv20_60",
    "log_rv252_60",
    "log_ewma_60",
    "log_yz22_60",
    "log_yz5_60",
    "log_yz_chg_5",
    "log_yz_chg_20",
    "log_park20_60",
    "log_gk20_60",
    "log_rs20_60",
    "log_overnight_share",
    "z_ret_1d",
    "z_ret_5d",
    "z_ret_20d",
    "z_downside_1d",
    "z_upside_1d",
    "z_downside_5d_mean",
    "neg_indicator",
    "log_har_d_60",
    "log_har_w_60",
    "log_har_m_60",
    "rel_vol_of_vol_20",
)


def _safe_log_ratio(num: np.ndarray, den: np.ndarray, *, clip_abs: float = 8.0) -> np.ndarray:
    valid = np.isfinite(num) & np.isfinite(den)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(np.maximum(num, EPS_VAR) / np.maximum(den, EPS_VAR))
    return np.where(valid, np.clip(ratio, -clip_abs, clip_abs), np.nan)


def build_g3_v2_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build strictly causal, scale-invariant 24-column G3 v2 feature block."""
    open_ = frame["Open"].to_numpy(dtype=float)
    high = frame["High"].to_numpy(dtype=float)
    low = frame["Low"].to_numpy(dtype=float)
    close = frame["Close"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.log(close / np.roll(close, 1))
    logret[0] = np.nan
    sq = logret**2
    s_ret = pd.Series(logret, index=frame.index)
    s_sq = pd.Series(sq, index=frame.index)

    v60 = s_ret.rolling(60, min_periods=60).var(ddof=1).to_numpy()
    safe_v60 = np.where(np.isfinite(v60), np.maximum(v60, EPS_VAR), np.nan)
    sig60 = np.sqrt(safe_v60)
    log_v60 = np.log(safe_v60)

    rv5 = s_sq.rolling(5, min_periods=5).mean().to_numpy()
    rv10 = s_sq.rolling(10, min_periods=10).mean().to_numpy()
    rv20 = s_sq.rolling(20, min_periods=20).mean().to_numpy()
    rv252 = s_ret.rolling(252, min_periods=60).var(ddof=1).to_numpy()
    ewma_var = s_sq.ewm(alpha=0.06, adjust=False).mean().to_numpy()

    park, gk, rs = _range_variances(open_, high, low, close)
    yz = _yang_zhang_var(open_, high, low, close, window=22)
    s_yz = pd.Series(yz, index=frame.index)
    yz5 = s_yz.rolling(5, min_periods=5).mean().to_numpy()
    yz20 = s_yz.rolling(20, min_periods=20).mean().to_numpy()
    park20 = pd.Series(park, index=frame.index).rolling(20, min_periods=20).mean().to_numpy()
    gk20 = pd.Series(gk, index=frame.index).rolling(20, min_periods=20).mean().to_numpy()
    rs20 = pd.Series(rs, index=frame.index).rolling(20, min_periods=20).mean().to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        overnight = np.log(open_ / np.roll(close, 1))
    overnight[0] = np.nan
    overnight_var_20 = (
        pd.Series(overnight, index=frame.index).rolling(20, min_periods=20).var(ddof=1).to_numpy()
    )

    har_rv = sq.copy()
    if len(har_rv):
        har_rv[0] = 0.0
    har_d, har_w, har_m = _har_log_components(har_rv)

    c_ser = pd.Series(close, index=frame.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_5d = np.log(c_ser / c_ser.shift(5)).to_numpy()
        ret_20d = np.log(c_ser / c_ser.shift(20)).to_numpy()

    z1 = np.clip(logret / sig60, -10.0, 10.0)
    z5 = np.clip(ret_5d / (np.sqrt(5.0) * sig60), -10.0, 10.0)
    z20 = np.clip(ret_20d / (np.sqrt(20.0) * sig60), -10.0, 10.0)
    z_down_1d = np.where(np.isfinite(z1), np.where(z1 < 0.0, z1**2, 0.0), np.nan)
    z_up_1d = np.where(np.isfinite(z1), np.where(z1 > 0.0, z1**2, 0.0), np.nan)
    z_down_5d_mean = (
        pd.Series(z_down_1d, index=frame.index).rolling(5, min_periods=5).mean().to_numpy()
    )
    neg_ind = np.where(np.isfinite(logret), np.where(logret < 0.0, 1.0, 0.0), np.nan)
    vol_of_vol_20 = (
        pd.Series(np.sqrt(np.maximum(sq, 0.0))).rolling(20, min_periods=20).std(ddof=1).to_numpy()
    )

    out = pd.DataFrame(index=frame.index)
    out["log_rv5_60"] = _safe_log_ratio(rv5, safe_v60)
    out["log_rv10_60"] = _safe_log_ratio(rv10, safe_v60)
    out["log_rv20_60"] = _safe_log_ratio(rv20, safe_v60)
    out["log_rv252_60"] = _safe_log_ratio(rv252, safe_v60)
    out["log_ewma_60"] = _safe_log_ratio(ewma_var, safe_v60)
    out["log_yz22_60"] = _safe_log_ratio(yz, safe_v60)
    out["log_yz5_60"] = _safe_log_ratio(yz5, safe_v60)
    out["log_yz_chg_5"] = _safe_log_ratio(yz, yz5)
    out["log_yz_chg_20"] = _safe_log_ratio(yz, yz20)
    out["log_park20_60"] = _safe_log_ratio(park20, safe_v60)
    out["log_gk20_60"] = _safe_log_ratio(gk20, safe_v60)
    out["log_rs20_60"] = _safe_log_ratio(rs20, safe_v60)
    out["log_overnight_share"] = _safe_log_ratio(overnight_var_20, yz)
    out["z_ret_1d"] = z1
    out["z_ret_5d"] = z5
    out["z_ret_20d"] = z20
    out["z_downside_1d"] = z_down_1d
    out["z_upside_1d"] = z_up_1d
    out["z_downside_5d_mean"] = z_down_5d_mean
    out["neg_indicator"] = neg_ind
    out["log_har_d_60"] = np.clip(har_d - log_v60, -12.0, 8.0)
    out["log_har_w_60"] = np.clip(har_w - log_v60, -8.0, 8.0)
    out["log_har_m_60"] = np.clip(har_m - log_v60, -8.0, 8.0)
    out["rel_vol_of_vol_20"] = np.clip(vol_of_vol_20 / sig60, 0.0, 10.0)
    return out[list(G3_V2_FEATURES)]


def panel_rolling_base_60d(frame: pd.DataFrame, horizon: int) -> float:
    """Production 60-day close-to-close cumulative variance anchor ``H * Var_60d``."""
    close = frame["Close"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.log(close / np.roll(close, 1))
    window = logret[-60:]
    if len(window) < 60 or not np.isfinite(window).all():
        raise ValueError("insufficient history for 60d rolling base")
    return float(np.var(window, ddof=1) * horizon)
