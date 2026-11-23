"""Milestone 05 Empirical ML Validation, Calibration Gate & Report Generator.

Evaluates:
1. Exact locked shipped G3 ONNX artifacts (``backend/volatility_models/g3_h{5,10,20}.onnx``)
   on strictly purged out-of-sample data (``origin_date >= reserve_start = 2023-11-16`` and
   ``target_end_date <= as_of``) across the predeclared 30-ticker US + LSE panel
   (``TRI_EXCHANGE_TICKERS``) against production 60d ``rolling_mean`` and ``har_rv``
   (``causal_log_har``), with 20d ``G0`` (``persistence``) as a diagnostic reference.
2. Concrete step-by-step MSFT 5d/10d/20d snapshot trace decomposing the divergence between
   ``gpu_g3``, ``rolling_mean`` (60d), ``persistence`` (20d G0), and ``har_rv``.
3. Walk-forward evaluation of the exact served ``simple_forecast.py`` Ridge predictor and
   uncertainty band with strict target maturity (zero leakage), reporting both pointwise
   marginal and simultaneous 7-day path coverage against ``[75%, 85%]`` across calm vs
   elevated volatility regimes and US vs LSE exchanges, alongside scale-independent
   persistence MAE ratio significance and target-matured online conformal diagnostics.

Every reported statistic is computed directly from real normalized market observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from scipy.stats import f as f_dist

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend"), str(REPO_ROOT / "research")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from calendars import latest_completed_trading_session  # noqa: E402
from market_data.normalization import (  # noqa: E402
    daily_frame_fingerprint,
    normalize_daily_bars_with_audit,
)
from market_data.yahoo import YahooProvider  # noqa: E402
from services import g3_volatility as g3  # noqa: E402
from services.live_volatility import build_live_volatility_forecast  # noqa: E402
from services.simple_forecast import (  # noqa: E402
    FORECAST_DAYS,
    MIN_HISTORY_ROWS,
    ForecastDataset,
    _clip_predictions,
    build_features,
    chronological_masks,
    get_ticker_meta,
)
from services.volatility_snapshot import (  # noqa: E402
    _build_volatility_inference_snapshot,
    build_features_v5,
    causal_log_har_forecasts,
    realized_variance_proxies,
)

from research.volatility_structure import panel as vol_panel  # noqa: E402
from research.volatility_structure.data_utils import TRI_EXCHANGE_TICKERS  # noqa: E402
from research.volatility_structure.evaluation import hac_mean  # noqa: E402

# ---------------------------------------------------------------------------
# Predeclared Validation Protocol Constants (frozen before inspecting results)
# ---------------------------------------------------------------------------
OOS_START_DATE = "2023-11-16"  # Exact reserve_start from g3_h{5,10,20}.meta.json
POST_FREEZE_START_DATE = "2026-09-08"  # Strictly after 2026-09-07 freeze & cache date
MIN_COMMON_ORIGINS_PER_ASSET = 250  # ~1 trading year minimum per asset
ASSET_WIN_RATE_THRESHOLD = 0.70  # Predeclared 70% breadth gate
SIGNIFICANCE_ALPHA = 0.05
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261123

# Joint Mincer-Zarnowitz equivalence bounds (annualized variance space)
MZ_INTERCEPT_REL_BOUND = 0.15  # |alpha / mean(v_realized)| <= 15%
MZ_SLOPE_BOUNDS = (0.80, 1.20)  # beta in [0.80, 1.20]
MZ_MEAN_RATIO_BOUNDS = (0.85, 1.15)  # mean(forecast) / mean(realized) in [0.85, 1.15]

# Price conformal calibration bounds (nominal 80% central interval)
PRICE_COVERAGE_BOUNDS = (0.75, 0.85)
PRICE_TAIL_MISS_BOUNDS = (0.05, 0.15)  # nominal 10% each tail
ONLINE_CONFORMAL_WINDOW = 250
ONLINE_CONFORMAL_MIN_MATURED = 60
ONLINE_CONFORMAL_ALPHA = 0.20
ELEVATED_VOL_RATIO_THRESHOLD = 1.15  # Matches live_volatility.py frozen threshold
TRADING_SESSIONS_PER_YEAR = 252.0


@dataclass(frozen=True)
class TickerDataRecord:
    symbol: str
    exchange: str
    currency: str
    frame: pd.DataFrame
    rows: int
    start_date: str
    end_date: str
    raw_source_fingerprint: str
    data_fingerprint: str
    repairs: tuple[dict[str, Any], ...]


# ---------------------------------------------------------------------------
# Pure Econometric & Statistical Gate Functions (unit-tested independently)
# ---------------------------------------------------------------------------
def bartlett_hac_summary(values: Any, bandwidth: int) -> dict[str, Any]:
    """Date-aggregated Bartlett HAC summary using research evaluation.hac_mean."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 3:
        raise ValueError("HAC requires at least 3 observations")
    eff_bw = min(int(bandwidth), len(arr) - 1)
    base = hac_mean(arr, eff_bw)
    mean = float(base["mean_loss_improvement"])
    se = float(base["se"])
    if se > 0.0:
        t_stat = mean / se
        p_one_sided = float(norm.sf(t_stat))
    else:
        t_stat = float(np.inf if mean > 0 else (-np.inf if mean < 0 else 0.0))
        p_one_sided = 0.0 if mean > 0 else (1.0 if mean < 0 else 0.5)
    return {
        "dates": int(base["dates"]),
        "bandwidth": int(eff_bw),
        "mean": mean,
        "se": se,
        "t_stat": float(t_stat),
        "ci95": [float(base["ci95"][0]), float(base["ci95"][1])],
        "p_two_sided": float(base["p_two_sided"]) if base["p_two_sided"] is not None else 0.0,
        "p_one_sided_gt_zero": p_one_sided,
    }


def circular_block_bootstrap(
    values: Any,
    block_length: int,
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Circular date-block bootstrap for the mean of a time-ordered series."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < 3 or not np.isfinite(arr).all():
        raise ValueError("Circular block bootstrap requires >= 3 finite values")
    b = max(1, min(int(block_length), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(b, dtype=np.int64)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    idx = idx.reshape(n_boot, n_blocks * b)[:, :n]
    boot_means = arr[idx].mean(axis=1)
    obs_mean = float(arr.mean())
    ci_low = float(np.quantile(boot_means, 0.025))
    ci_high = float(np.quantile(boot_means, 0.975))
    se = float(np.std(boot_means, ddof=1))
    # Centered bootstrap two-sided p-value and one-sided probability of <= 0
    centered = boot_means - obs_mean
    p_two_sided = float((np.sum(np.abs(centered) >= abs(obs_mean)) + 1.0) / (n_boot + 1.0))
    p_one_sided_gt_zero = float((np.sum(boot_means <= 0.0) + 1.0) / (n_boot + 1.0))
    return {
        "dates": n,
        "block_length": b,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "mean": obs_mean,
        "boot_se": se,
        "ci95": [ci_low, ci_high],
        "p_two_sided": p_two_sided,
        "p_one_sided_gt_zero": p_one_sided_gt_zero,
    }


def circular_block_bootstrap_ratio(
    numerator: Any,
    denominator: Any,
    block_length: int,
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Circular date-block bootstrap for ratio mean(num) / mean(den)."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    n = len(num)
    if n != len(den) or n < 3 or not (np.isfinite(num).all() and np.isfinite(den).all()):
        raise ValueError("Invalid numerator/denominator arrays for ratio bootstrap")
    if float(den.mean()) <= 0.0:
        raise ValueError("Denominator mean must be strictly positive")
    b = max(1, min(int(block_length), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(b, dtype=np.int64)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    idx = idx.reshape(n_boot, n_blocks * b)[:, :n]
    num_means = num[idx].mean(axis=1)
    den_means = np.maximum(den[idx].mean(axis=1), 1e-12)
    ratios = num_means / den_means
    obs_ratio = float(num.mean() / den.mean())
    p_lt_one = float((np.sum(ratios >= 1.0) + 1.0) / (n_boot + 1.0))
    return {
        "dates": n,
        "block_length": b,
        "n_boot": int(n_boot),
        "ratio": obs_ratio,
        "ci95": [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))],
        "p_one_sided_lt_one": p_lt_one,
    }


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Step-down Holm-Bonferroni adjusted p-values preserving input order."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda pair: float(pair[1]))
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        scaled = min(1.0, float(p_val) * (m - rank))
        running_max = max(running_max, scaled)
        adjusted[orig_idx] = float(running_max)
    return adjusted


def mincer_zarnowitz_panel_test(
    realized_ann_var: Any,
    forecast_ann_var: Any,
    dates: Any,
    *,
    bandwidth: int,
    intercept_rel_bound: float = MZ_INTERCEPT_REL_BOUND,
    slope_bounds: tuple[float, float] = MZ_SLOPE_BOUNDS,
    mean_ratio_bounds: tuple[float, float] = MZ_MEAN_RATIO_BOUNDS,
    alpha_sig: float = SIGNIFICANCE_ALPHA,
) -> dict[str, Any]:
    """Joint Mincer-Zarnowitz Wald test of (intercept, slope) = (0, 1) with Driscoll-Kraay HAC.

    Regresses annualized realized variance ``y`` on ``[1, f]``. Score vectors are summed
    within each calendar date and weighted across lags with a Bartlett kernel so both
    cross-sectional dependence and overlapping-horizon autocorrelation are accounted for.
    Explicit equivalence bounds on BOTH normalized intercept ``alpha / mean(y)`` and slope
    ``beta`` (plus mean forecast/realized ratio) catch level bias even when slope ~= 1.
    """
    y = np.asarray(realized_ann_var, dtype=np.float64)
    f = np.asarray(forecast_ann_var, dtype=np.float64)
    d = pd.to_datetime(dates).to_numpy()
    if len(y) != len(f) or len(y) != len(d) or len(y) < 10:
        raise ValueError("Mincer-Zarnowitz requires >= 10 aligned observations")
    if not (np.isfinite(y).all() and np.isfinite(f).all()):
        raise ValueError("Mincer-Zarnowitz inputs must be finite")

    mean_y = float(np.mean(y))
    mean_f = float(np.mean(f))
    if mean_y <= 0.0:
        raise ValueError("Mean realized variance must be positive")

    n = len(y)
    x = np.column_stack([np.ones(n, dtype=np.float64), f])
    xtx = x.T @ x
    if np.linalg.cond(xtx) > 1e14:
        raise ValueError("Singular Mincer-Zarnowitz design matrix")
    theta = np.linalg.solve(xtx, x.T @ y)
    alpha_hat, beta_hat = float(theta[0]), float(theta[1])
    resid = y - x @ theta
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - mean_y) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    # Aggregate observation scores by ordered calendar date (Driscoll-Kraay clustering)
    scores = x * resid[:, None]
    score_df = pd.DataFrame({"date": d, "s0": scores[:, 0], "s1": scores[:, 1]})
    daily_scores = (
        score_df.groupby("date", sort=True)[["s0", "s1"]].sum().to_numpy(dtype=np.float64)
    )
    t_dates = len(daily_scores)
    if t_dates < 4:
        raise ValueError("Mincer-Zarnowitz requires at least 4 distinct dates")

    eff_bw = min(max(int(bandwidth), 0), t_dates - 1)
    omega = daily_scores.T @ daily_scores
    for lag in range(1, eff_bw + 1):
        weight = 1.0 - lag / (eff_bw + 1.0)
        gamma = daily_scores[lag:].T @ daily_scores[:-lag]
        omega = omega + weight * (gamma + gamma.T)

    xtx_inv = np.linalg.inv(xtx)
    dof_scale = float(t_dates) / max(t_dates - 1, 1)
    cov = dof_scale * (xtx_inv @ omega @ xtx_inv)
    alpha_se = float(np.sqrt(max(cov[0, 0], 0.0)))
    beta_se = float(np.sqrt(max(cov[1, 1], 0.0)))

    diff = np.array([alpha_hat - 0.0, beta_hat - 1.0], dtype=np.float64)
    try:
        wald_chi2 = float(diff @ np.linalg.solve(cov, diff))
    except np.linalg.LinAlgError:
        wald_chi2 = float(diff @ np.linalg.pinv(cov) @ diff)
    wald_chi2 = max(wald_chi2, 0.0)
    p_wald_chi2 = float(chi2.sf(wald_chi2, df=2))
    wald_f = wald_chi2 / 2.0
    p_wald_f = float(f_dist.sf(wald_f, dfn=2, dfd=max(t_dates - 2, 1)))

    z95 = 1.959963984540054
    alpha_ci95 = [alpha_hat - z95 * alpha_se, alpha_hat + z95 * alpha_se]
    beta_ci95 = [beta_hat - z95 * beta_se, beta_hat + z95 * beta_se]
    alpha_rel = alpha_hat / mean_y
    alpha_rel_se = alpha_se / mean_y
    alpha_rel_ci95 = [alpha_rel - z95 * alpha_rel_se, alpha_rel + z95 * alpha_rel_se]
    mean_ratio = mean_f / mean_y

    intercept_point_ok = abs(alpha_rel) <= intercept_rel_bound
    intercept_ci_ok = (
        alpha_rel_ci95[0] >= -intercept_rel_bound and alpha_rel_ci95[1] <= intercept_rel_bound
    )
    slope_point_ok = slope_bounds[0] <= beta_hat <= slope_bounds[1]
    slope_ci_ok = beta_ci95[0] >= slope_bounds[0] and beta_ci95[1] <= slope_bounds[1]
    mean_ratio_ok = mean_ratio_bounds[0] <= mean_ratio <= mean_ratio_bounds[1]

    # Joint calibration gate: either fail to reject H0:(0,1) while point estimates & mean ratio
    # stay inside bounds, OR both 95% HAC CIs lie strictly inside predeclared equivalence bounds.
    wald_not_rejected = p_wald_chi2 >= alpha_sig
    mz_gate_pass = bool(
        (wald_not_rejected and intercept_point_ok and slope_point_ok and mean_ratio_ok)
        or (intercept_ci_ok and slope_ci_ok and mean_ratio_ok)
    )
    return {
        "n_obs": int(n),
        "n_dates": int(t_dates),
        "bandwidth": int(eff_bw),
        "mean_realized_ann_var": mean_y,
        "mean_forecast_ann_var": mean_f,
        "mean_forecast_to_realized_ratio": float(mean_ratio),
        "intercept": alpha_hat,
        "intercept_se": alpha_se,
        "intercept_ci95": [float(alpha_ci95[0]), float(alpha_ci95[1])],
        "intercept_rel": float(alpha_rel),
        "intercept_rel_se": float(alpha_rel_se),
        "intercept_rel_ci95": [float(alpha_rel_ci95[0]), float(alpha_rel_ci95[1])],
        "slope": beta_hat,
        "slope_se": beta_se,
        "slope_ci95": [float(beta_ci95[0]), float(beta_ci95[1])],
        "r2": float(r2),
        "wald_chi2": float(wald_chi2),
        "p_wald_chi2": p_wald_chi2,
        "wald_f": float(wald_f),
        "p_wald_f": p_wald_f,
        "bounds": {
            "intercept_rel_abs_max": float(intercept_rel_bound),
            "slope_bounds": [float(slope_bounds[0]), float(slope_bounds[1])],
            "mean_ratio_bounds": [float(mean_ratio_bounds[0]), float(mean_ratio_bounds[1])],
        },
        "checks": {
            "wald_not_rejected": wald_not_rejected,
            "intercept_point_in_bounds": intercept_point_ok,
            "intercept_ci_in_bounds": intercept_ci_ok,
            "slope_point_in_bounds": slope_point_ok,
            "slope_ci_in_bounds": slope_ci_ok,
            "mean_ratio_in_bounds": mean_ratio_ok,
        },
        "mz_gate_pass": mz_gate_pass,
    }


def evaluate_volatility_horizon_gate(
    *,
    vs_rolling: dict[str, Any],
    vs_har: dict[str, Any],
    us_vs_rolling: dict[str, Any],
    lse_vs_rolling: dict[str, Any],
    mz_g3: dict[str, Any],
    win_rate_threshold: float = ASSET_WIN_RATE_THRESHOLD,
    alpha_sig: float = SIGNIFICANCE_ALPHA,
) -> dict[str, Any]:
    """Apply predeclared 5-clause PROMOTE_G3_TO_AUTO binary gate for one horizon."""
    clauses = {
        "qlike_beats_rolling_60d_pooled_holm": bool(
            vs_rolling["mean_delta_qlike"] > 0.0
            and vs_rolling["hac_primary"]["p_holm"] < alpha_sig
            and vs_rolling["hac_primary"]["ci95"][0] > 0.0
            and vs_rolling["hac_2x"]["ci95"][0] > 0.0
            and vs_rolling["block_bootstrap"]["ci95"][0] > 0.0
        ),
        "qlike_beats_har_rv_pooled_holm": bool(
            vs_har["mean_delta_qlike"] > 0.0
            and vs_har["hac_primary"]["p_holm"] < alpha_sig
            and vs_har["hac_primary"]["ci95"][0] > 0.0
            and vs_har["hac_2x"]["ci95"][0] > 0.0
            and vs_har["block_bootstrap"]["ci95"][0] > 0.0
        ),
        "win_rate_ge_70pct_vs_rolling_pooled_us_lse": bool(
            vs_rolling["asset_win_rate"] >= win_rate_threshold
            and us_vs_rolling["asset_win_rate"] >= win_rate_threshold
            and lse_vs_rolling["asset_win_rate"] >= win_rate_threshold
            and vs_har["asset_win_rate"] >= win_rate_threshold
        ),
        "mincer_zarnowitz_joint_calibration": bool(mz_g3["mz_gate_pass"]),
        "subpanel_non_degradation_us_and_lse": bool(
            us_vs_rolling["mean_delta_qlike"] > 0.0
            and lse_vs_rolling["mean_delta_qlike"] > 0.0
            and us_vs_rolling["hac_primary"]["ci95"][1] > 0.0
            and lse_vs_rolling["hac_primary"]["ci95"][1] > 0.0
        ),
    }
    passed = all(clauses.values())
    return {
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "clauses": clauses,
        "failed_clauses": [name for name, ok in clauses.items() if not ok],
    }


def evaluate_price_band_gate(
    *,
    pooled_pointwise_coverage: float,
    per_horizon_coverages: dict[int, float],
    us_pointwise_coverage: float,
    lse_pointwise_coverage: float,
    calm_pointwise_coverage: float,
    elevated_pointwise_coverage: float,
    pooled_lower_tail_miss: float,
    pooled_upper_tail_miss: float,
    coverage_bounds: tuple[float, float] = PRICE_COVERAGE_BOUNDS,
    tail_bounds: tuple[float, float] = PRICE_TAIL_MISS_BOUNDS,
) -> dict[str, Any]:
    """Apply predeclared binary gate for UPGRADE_PRICE_BAND_LABEL_TO_CALIBRATED."""
    lo, hi = coverage_bounds
    t_lo, t_hi = tail_bounds
    clauses = {
        "pooled_pointwise_in_75_85": bool(lo <= pooled_pointwise_coverage <= hi),
        "all_horizons_1_to_7_in_75_85": bool(
            len(per_horizon_coverages) == FORECAST_DAYS
            and all(lo <= cov <= hi for cov in per_horizon_coverages.values())
        ),
        "us_and_lse_pointwise_in_75_85": bool(
            (lo <= us_pointwise_coverage <= hi) and (lo <= lse_pointwise_coverage <= hi)
        ),
        "calm_and_elevated_regimes_in_75_85": bool(
            (lo <= calm_pointwise_coverage <= hi) and (lo <= elevated_pointwise_coverage <= hi)
        ),
        "tail_symmetry_5_to_15_each_tail": bool(
            (t_lo <= pooled_lower_tail_miss <= t_hi) and (t_lo <= pooled_upper_tail_miss <= t_hi)
        ),
    }
    passed = all(clauses.values())
    return {
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "clauses": clauses,
        "failed_clauses": [name for name, ok in clauses.items() if not ok],
    }


def strict_matured_conformal_quantiles(
    past_residuals: np.ndarray,
    past_origin_positions: np.ndarray,
    current_origin_position: int,
    horizon_days: int,
    *,
    alpha: float = ONLINE_CONFORMAL_ALPHA,
    max_window: int = ONLINE_CONFORMAL_WINDOW,
    min_matured: int = ONLINE_CONFORMAL_MIN_MATURED,
) -> tuple[float, float, int] | None:
    """Return finite-sample split-conformal lower/upper quantiles using ONLY matured residuals.

    A residual from origin ``s`` for horizon ``H`` matures at ``s + H``; it is admitted
    strictly iff ``s + H <= current_origin_position``.
    """
    matured_mask = (past_origin_positions + int(horizon_days)) <= int(current_origin_position)
    matured = past_residuals[matured_mask]
    matured = matured[np.isfinite(matured)]
    if len(matured) < min_matured:
        return None
    if len(matured) > max_window:
        matured = matured[-max_window:]
    n = len(matured)
    sorted_vals = np.sort(matured)
    low_rank = max(0, int(np.floor((alpha / 2.0) * (n + 1))) - 1)
    high_rank = min(n - 1, int(np.ceil((1.0 - alpha / 2.0) * (n + 1))) - 1)
    return float(sorted_vals[low_rank]), float(sorted_vals[high_rank]), int(n)


# ---------------------------------------------------------------------------
# Step 1: Artifact Inspection & Checksum Verification
# ---------------------------------------------------------------------------
def verify_locked_g3_artifacts() -> dict[str, Any]:
    """Verify exact locked ONNX SHA-256 checksums, metadata, and absence of g3_h1.onnx."""
    model_dir = g3._model_dir()
    freeze_manifest_path = REPO_ROOT / "artifacts" / "g3_panel_v1" / "manifest.json"
    freeze_manifest = (
        json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
        if freeze_manifest_path.is_file()
        else None
    )
    freeze_manifest_sha = (
        hashlib.sha256(freeze_manifest_path.read_bytes()).hexdigest()
        if freeze_manifest_path.is_file()
        else None
    )

    horizons_report: dict[str, Any] = {}
    for horizon in g3.G3_HORIZONS:
        onnx_path = model_dir / f"g3_h{horizon}.onnx"
        meta = g3._load_metadata(horizon)
        actual_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        if actual_sha != meta["onnx_sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {onnx_path}: {actual_sha} != {meta['onnx_sha256']}"
            )
        horizons_report[str(horizon)] = {
            "onnx_path": str(onnx_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "onnx_size_bytes": int(onnx_path.stat().st_size),
            "onnx_sha256": actual_sha,
            "checksum_verified": True,
            "booster_sha256": meta["booster_sha256"],
            "freeze_manifest_sha256": meta["freeze_manifest_sha256"],
            "freeze_manifest_matches_disk": bool(
                freeze_manifest_sha == meta["freeze_manifest_sha256"]
            ),
            "serving_contract": meta["serving_contract"],
            "graph_base_score": float(meta["graph_base_score"]),
            "objective": meta["objective"],
            "train_rows": int(meta["train_rows"]),
            "validation_start": meta["validation_start"],
            "reserve_start": meta["reserve_start"],
            "n_features": len(meta["features"]),
            "features_match_serving_code": bool(meta["features"] == list(g3.G3_FEATURES)),
            "inference_formula": meta["inference"],
        }

    return {
        "model_dir": str(model_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "g3_model_version": g3.G3_MODEL_VERSION,
        "g3_feature_set_version": g3.G3_FEATURE_SET_VERSION,
        "g3_h1_onnx_exists": (model_dir / "g3_h1.onnx").is_file(),
        "h1_policy_preserved": "garch_11",
        "freeze_created_at": freeze_manifest.get("created_at") if freeze_manifest else None,
        "horizons": horizons_report,
    }


# ---------------------------------------------------------------------------
# Step 2: Panel Loading & M2 Auditable Normalization
# ---------------------------------------------------------------------------
def load_or_fetch_panel(
    *,
    tickers: tuple[str, ...] = TRI_EXCHANGE_TICKERS,
    as_of: str | None = None,
    years: int = 8,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> tuple[list[TickerDataRecord], list[dict[str, Any]], str]:
    """Load normalized OHLCV frames from cache or live Yahoo via M2 normalization."""
    if as_of is None:
        us_sess = latest_completed_trading_session(exchange="NYSE").date().isoformat()
        uk_sess = latest_completed_trading_session(exchange="LSE").date().isoformat()
        as_of = min(us_sess, uk_sess)

    cached_meta: dict[str, Any] = {}
    meta_path = cache_dir / "panel_meta.json" if cache_dir is not None else None
    if not refresh and meta_path is not None and meta_path.is_file():
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        if loaded.get("as_of") == as_of:
            cached_meta = loaded.get("tickers", {})

    provider = YahooProvider()
    records: list[TickerDataRecord] = []
    attrition: list[dict[str, Any]] = []
    new_meta: dict[str, Any] = {}

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for sym in tickers:
        meta_info = get_ticker_meta(sym)
        exchange = "LSE" if sym.endswith(".L") else "US"
        parquet_path = cache_dir / f"{sym}.parquet" if cache_dir is not None else None
        if (
            not refresh
            and parquet_path is not None
            and parquet_path.is_file()
            and sym in cached_meta
        ):
            df = pd.read_parquet(parquet_path)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()
            info = cached_meta[sym]
            df.attrs.update(
                {
                    "data_provider": "yahoo",
                    "data_as_of": info["end"],
                    "data_feed": "yahoo_chart",
                    "price_adjustment": "auto",
                    "data_fingerprint": info["data_fingerprint"],
                    "raw_source_fingerprint": info["raw_source_fingerprint"],
                }
            )
            records.append(
                TickerDataRecord(
                    symbol=sym,
                    exchange=exchange,
                    currency=meta_info["currency"],
                    frame=df,
                    rows=int(len(df)),
                    start_date=info["start"],
                    end_date=info["end"],
                    raw_source_fingerprint=info["raw_source_fingerprint"],
                    data_fingerprint=info["data_fingerprint"],
                    repairs=tuple(info.get("repairs", ())),
                )
            )
            new_meta[sym] = info
            continue

        try:
            raw = provider.fetch_raw_daily_bars(sym, years=years)
            norm_df, raw_fp, audit = normalize_daily_bars_with_audit(
                raw.frame,
                provider="yahoo",
                symbol=sym,
                timestamp_kind=raw.timestamp_kind,
                max_session=as_of,
            )
            norm_df.attrs.update(
                {
                    "data_provider": "yahoo",
                    "data_feed": raw.feed,
                    "price_adjustment": raw.adjustment,
                    "data_as_of": norm_df.index[-1].date().isoformat(),
                }
            )
            fp = daily_frame_fingerprint(norm_df)
            norm_df.attrs["data_fingerprint"] = fp
            repairs = tuple(r.to_dict() for r in audit)
            s0 = norm_df.index[0].date().isoformat()
            s1 = norm_df.index[-1].date().isoformat()
            info = {
                "rows": int(len(norm_df)),
                "start": s0,
                "end": s1,
                "as_of": as_of,
                "repairs": list(repairs),
                "raw_source_fingerprint": raw_fp,
                "data_fingerprint": fp,
            }
            if parquet_path is not None:
                clean = norm_df.copy()
                clean.attrs = {}
                clean.to_parquet(parquet_path)
            records.append(
                TickerDataRecord(
                    symbol=sym,
                    exchange=exchange,
                    currency=meta_info["currency"],
                    frame=norm_df,
                    rows=int(len(norm_df)),
                    start_date=s0,
                    end_date=s1,
                    raw_source_fingerprint=raw_fp,
                    data_fingerprint=fp,
                    repairs=repairs,
                )
            )
            new_meta[sym] = info
        except Exception as exc:
            attrition.append(
                {
                    "symbol": sym,
                    "exchange": exchange,
                    "stage": "fetch_or_m2_normalization",
                    "reason": str(exc),
                }
            )

    if meta_path is not None and new_meta:
        meta_path.write_text(
            json.dumps(
                {
                    "as_of": as_of,
                    "ok": [r.symbol for r in records],
                    "failed": attrition,
                    "tickers": new_meta,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return records, attrition, as_of


# ---------------------------------------------------------------------------
# Step 3: Concrete MSFT 5d/10d/20d Discrepancy Trace
# ---------------------------------------------------------------------------
def trace_msft_snapshot(msft_record: TickerDataRecord) -> dict[str, Any]:
    """Step-by-step mathematical decomposition of MSFT G3 vs rolling_mean vs har_rv."""
    df = msft_record.frame
    snap = _build_volatility_inference_snapshot("MSFT", df)
    feats = g3.build_g3_features(df)
    latest_row = feats.iloc[[-1]].to_numpy(dtype=np.float32)

    horizons_trace: dict[str, Any] = {}
    for horizon in g3.G3_HORIZONS:
        models_out: dict[str, Any] = {}
        for model_name in (
            "rolling_mean",
            "persistence",
            "har_rv",
            "ewma",
            "garch_11",
            "gpu_g3",
        ):
            payload = build_live_volatility_forecast(snap, horizon=horizon, model=model_name)
            fc = payload["forecast"]
            ev = payload["evidence"]
            models_out[model_name] = {
                "cumulative_variance": float(fc["expected_cumulative_variance"]),
                "annualized_volatility": float(fc["expected_annualized_volatility"]),
                "risk_level": ev["risk_level"],
                "risk_ratio_vs_trailing_60d": float(ev["risk_ratio_vs_trailing_60d"]),
            }

        meta = g3._load_metadata(horizon)
        session = g3._load_session(horizon)
        raw_onnx = float(np.asarray(session.run(None, {"input": latest_row})[0]).reshape(-1)[0])
        graph_base_score = float(meta["graph_base_score"])
        delta = raw_onnx - graph_base_score
        base_20d = g3.panel_rolling_base(df, horizon)
        unclipped_log_var = delta + float(np.log(max(base_20d, 1e-12)))
        clipped_log_var = float(np.clip(unclipped_log_var, -30.0, 10.0))
        g3_var = float(np.exp(clipped_log_var))
        g3_ann = float(np.sqrt(g3_var / horizon * TRADING_SESSIONS_PER_YEAR))

        roll60_var = models_out["rolling_mean"]["cumulative_variance"]
        roll60_ann = models_out["rolling_mean"]["annualized_volatility"]
        base20_ann = models_out["persistence"]["annualized_volatility"]

        var_ratio_20_vs_60 = base_20d / roll60_var
        vol_ratio_20_vs_60 = base20_ann / roll60_ann
        var_mult_g3 = float(np.exp(delta))
        vol_mult_g3 = float(np.exp(0.5 * delta))
        total_var_ratio = g3_var / roll60_var
        total_vol_ratio = g3_ann / roll60_ann

        log_vol_shift_total = float(np.log(total_vol_ratio))
        log_vol_shift_base = float(np.log(vol_ratio_20_vs_60))
        log_vol_shift_delta = 0.5 * delta

        horizons_trace[str(horizon)] = {
            "models": models_out,
            "decomposition": {
                "rolling_60d_cum_var": roll60_var,
                "rolling_60d_ann_vol": roll60_ann,
                "g0_20d_cum_var": base_20d,
                "g0_20d_ann_vol": base20_ann,
                "log_g0_20d_cum_var": float(np.log(max(base_20d, 1e-12))),
                "onnx_raw_output": raw_onnx,
                "graph_base_score": graph_base_score,
                "learned_log_var_delta": delta,
                "learned_var_multiplier": var_mult_g3,
                "learned_vol_multiplier": vol_mult_g3,
                "unclipped_log_var": unclipped_log_var,
                "clip_bounds": [-30.0, 10.0],
                "clip_binding": bool(unclipped_log_var != clipped_log_var),
                "g3_final_cum_var": g3_var,
                "g3_final_ann_vol": g3_ann,
                "step1_base_var_ratio_20_over_60": float(var_ratio_20_vs_60),
                "step1_base_vol_ratio_20_over_60": float(vol_ratio_20_vs_60),
                "step1_base_ann_vol_delta_pp": float((base20_ann - roll60_ann) * 100.0),
                "step2_g3_ann_vol_delta_pp": float((g3_ann - base20_ann) * 100.0),
                "total_ann_vol_delta_pp": float((g3_ann - roll60_ann) * 100.0),
                "total_var_ratio_g3_over_60": float(total_var_ratio),
                "total_vol_ratio_g3_over_60": float(total_vol_ratio),
                "log_vol_share_base_20_vs_60": float(log_vol_shift_base / log_vol_shift_total)
                if abs(log_vol_shift_total) > 1e-12
                else 0.0,
                "log_vol_share_learned_delta": float(log_vol_shift_delta / log_vol_shift_total)
                if abs(log_vol_shift_total) > 1e-12
                else 0.0,
                "accounting_identity_abs_error": float(
                    abs((vol_ratio_20_vs_60 * vol_mult_g3) - total_vol_ratio)
                ),
            },
        }

    # Trailing 5-session trajectory for H=5 showing reactivity
    close = df["Close"].to_numpy(dtype=np.float64)
    logret = np.empty_like(close)
    logret[0] = np.nan
    logret[1:] = np.log(close[1:] / close[:-1])
    sess5 = g3._load_session(5)
    bs5 = float(g3._load_metadata(5)["graph_base_score"])
    trajectory = []
    for idx in range(len(df) - 5, len(df)):
        row_f32 = feats.iloc[[idx]].to_numpy(dtype=np.float32)
        raw5 = float(np.asarray(sess5.run(None, {"input": row_f32})[0]).reshape(-1)[0])
        delta5 = raw5 - bs5
        b20 = float(np.var(logret[idx - 19 : idx + 1], ddof=1) * 5.0)
        b60 = float(np.var(logret[idx - 59 : idx + 1], ddof=1) * 5.0)
        g3_v = float(np.exp(np.clip(delta5 + np.log(max(b20, 1e-12)), -30.0, 10.0)))
        trajectory.append(
            {
                "date": df.index[idx].date().isoformat(),
                "close": float(close[idx]),
                "ret_1d": float(logret[idx]),
                "rolling_60d_ann_vol": float(np.sqrt(b60 / 5.0 * TRADING_SESSIONS_PER_YEAR)),
                "g0_20d_ann_vol": float(np.sqrt(b20 / 5.0 * TRADING_SESSIONS_PER_YEAR)),
                "delta_h5": float(delta5),
                "g3_h5_ann_vol": float(np.sqrt(g3_v / 5.0 * TRADING_SESSIONS_PER_YEAR)),
            }
        )

    feature_summary = []
    for col in g3.G3_FEATURES:
        val = float(feats[col].iloc[-1])
        trail = feats[col].iloc[-252:]
        feature_summary.append(
            {
                "feature": col,
                "value": val,
                "trailing_1y_median": float(trail.median()),
                "trailing_1y_percentile": float((trail <= val).mean() * 100.0),
            }
        )

    return {
        "ticker": "MSFT",
        "origin_date": snap.origin_date,
        "origin_close": float(snap.origin_close),
        "data_provider": "yahoo",
        "data_fingerprint": msft_record.data_fingerprint,
        "horizons": horizons_trace,
        "recent_h5_trajectory": trajectory,
        "latest_features": feature_summary,
    }


# ---------------------------------------------------------------------------
# Step 4: Volatility Promotion Gate Evaluation
# ---------------------------------------------------------------------------
def _score_volatility_comparator(
    df_h: pd.DataFrame,
    *,
    horizon: int,
    cand_col: str,
    base_col: str,
    min_origins: int = MIN_COMMON_ORIGINS_PER_ASSET,
) -> dict[str, Any]:
    """Compute paired QLIKE comparison, HAC (L & 2L), block bootstrap, and asset win rates."""
    qlike_cand = np.array(
        [vol_panel.qlike(y, f) for y, f in zip(df_h["realized"], df_h[cand_col], strict=True)],
        dtype=np.float64,
    )
    qlike_base = np.array(
        [vol_panel.qlike(y, f) for y, f in zip(df_h["realized"], df_h[base_col], strict=True)],
        dtype=np.float64,
    )
    diff = qlike_base - qlike_cand  # positive favors candidate (G3)
    scored = df_h[["symbol", "exchange", "origin_date", "realized", cand_col, base_col]].copy()
    scored["qlike_cand"] = qlike_cand
    scored["qlike_base"] = qlike_base
    scored["diff"] = diff

    daily = scored.groupby("origin_date", sort=True)["diff"].mean()
    lag_primary = horizon - 1
    lag_2x = 2 * (horizon - 1)
    hac_primary = bartlett_hac_summary(daily.to_numpy(), lag_primary)
    hac_2x = bartlett_hac_summary(daily.to_numpy(), lag_2x)
    block_boot = circular_block_bootstrap(
        daily.to_numpy(),
        block_length=2 * horizon,
        n_boot=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED + horizon,
    )

    per_asset = (
        scored.groupby("symbol", sort=True)
        .agg(
            exchange=("exchange", "first"),
            common_origins=("diff", "size"),
            mean_qlike_base=("qlike_base", "mean"),
            mean_qlike_cand=("qlike_cand", "mean"),
            mean_delta_qlike=("diff", "mean"),
        )
        .reset_index()
    )
    eligible = per_asset[per_asset["common_origins"] >= min_origins]
    n_eligible = int(len(eligible))
    n_wins = int((eligible["mean_delta_qlike"] > 0.0).sum())
    win_rate = float(n_wins / n_eligible) if n_eligible > 0 else 0.0

    mean_base = float(scored["qlike_base"].mean())
    mean_cand = float(scored["qlike_cand"].mean())
    mean_delta = float(scored["diff"].mean())
    rel_improv = float(mean_delta / mean_base) if mean_base > 0.0 else 0.0

    return {
        "common_origins": int(len(scored)),
        "common_dates": int(len(daily)),
        "common_assets_total": int(len(per_asset)),
        "eligible_assets_denominator": n_eligible,
        "min_origins_threshold": int(min_origins),
        "assets_won": n_wins,
        "asset_win_rate": win_rate,
        "dates_won_fraction": float((daily > 0.0).mean()) if len(daily) else 0.0,
        "mean_qlike_base": mean_base,
        "mean_qlike_cand": mean_cand,
        "median_qlike_base": float(scored["qlike_base"].median()),
        "median_qlike_cand": float(scored["qlike_cand"].median()),
        "mean_delta_qlike": mean_delta,
        "date_aggregated_mean_delta_qlike": float(daily.mean()),
        "relative_improvement": rel_improv,
        "hac_primary": hac_primary,
        "hac_2x": hac_2x,
        "block_bootstrap": block_boot,
        "per_asset": per_asset.to_dict(orient="records"),
    }


def evaluate_volatility_panel(
    records: list[TickerDataRecord],
    *,
    oos_start: str = OOS_START_DATE,
    as_of: str,
) -> dict[str, Any]:
    """Evaluate locked G3 ONNX artifacts against 60d rolling_mean, har_rv, and 20d G0."""
    oos_start_ts = pd.Timestamp(oos_start)
    as_of_ts = pd.Timestamp(as_of)
    post_freeze_ts = pd.Timestamp(POST_FREEZE_START_DATE)

    rows_by_h: dict[int, list[pd.DataFrame]] = {h: [] for h in g3.G3_HORIZONS}
    per_ticker_coverage: list[dict[str, Any]] = []
    parity_checks: dict[int, float] = {h: 0.0 for h in g3.G3_HORIZONS}

    sessions = {h: g3._load_session(h) for h in g3.G3_HORIZONS}
    base_scores = {h: float(g3._load_metadata(h)["graph_base_score"]) for h in g3.G3_HORIZONS}

    for rec in records:
        df = rec.frame
        feats_g3 = g3.build_g3_features(df)
        feats_v5 = build_features_v5(df)
        proxy = realized_variance_proxies(df)["RV_C2C"].clip(lower=1e-12)
        har_all = causal_log_har_forecasts(proxy, horizons=g3.G3_HORIZONS)

        close = df["Close"].to_numpy(dtype=np.float64)
        logret = np.empty_like(close)
        logret[0] = np.nan
        logret[1:] = np.log(close[1:] / close[:-1])
        roll20_var = (
            pd.Series(logret, index=df.index).rolling(20, min_periods=20).var(ddof=1).to_numpy()
        )
        roll60_var = feats_v5["Vol_C2C_60"].to_numpy(dtype=np.float64) ** 2

        x_mat = feats_g3.to_numpy(dtype=np.float32)
        finite_x = np.isfinite(x_mat).all(axis=1)
        idx_dates = df.index
        ticker_cov: dict[str, Any] = {
            "symbol": rec.symbol,
            "exchange": rec.exchange,
            "currency": rec.currency,
            "total_rows": rec.rows,
            "start_date": rec.start_date,
            "end_date": rec.end_date,
            "m2_repairs": len(rec.repairs),
        }

        for h_idx, horizon in enumerate(g3.G3_HORIZONS):
            y_fwd = vol_panel.forward_realized_variance(df["Close"], horizon).to_numpy(
                dtype=np.float64
            )
            pos = np.arange(len(df), dtype=np.int64)
            end_pos = np.minimum(pos + horizon, len(df) - 1)
            target_end_dates = idx_dates[end_pos]
            in_bounds = (pos + horizon) < len(df)

            base20_h = roll20_var * horizon
            roll60_h = np.maximum(roll60_var * horizon, 1e-12)
            har_h = har_all[:, h_idx]

            g3_h = np.full(len(df), np.nan, dtype=np.float64)
            valid_g3 = finite_x & np.isfinite(base20_h) & (base20_h > 0.0)
            if valid_g3.any():
                raw_out = (
                    np.asarray(sessions[horizon].run(None, {"input": x_mat[valid_g3]})[0])
                    .reshape(-1)
                    .astype(np.float64)
                )
                log_v = np.clip(
                    raw_out - base_scores[horizon] + np.log(np.maximum(base20_h[valid_g3], 1e-12)),
                    -30.0,
                    10.0,
                )
                g3_h[valid_g3] = np.exp(log_v)

            # Verify exact parity with backend g3.g3_cumulative_variance on latest session
            if np.isfinite(g3_h[-1]):
                ref_last = g3.g3_cumulative_variance(df, horizon)
                rel_err = abs(g3_h[-1] - ref_last) / max(abs(ref_last), 1e-12)
                parity_checks[horizon] = max(parity_checks[horizon], float(rel_err))

            oos_mask = (
                (idx_dates >= oos_start_ts)
                & in_bounds
                & (target_end_dates <= as_of_ts)
                & np.isfinite(y_fwd)
                & (y_fwd >= 0.0)
                & np.isfinite(g3_h)
                & (g3_h > 0.0)
                & np.isfinite(roll60_h)
                & (roll60_h > 0.0)
                & np.isfinite(har_h)
                & (har_h > 0.0)
                & np.isfinite(base20_h)
                & (base20_h > 0.0)
            )
            ticker_cov[f"common_origins_h{horizon}"] = int(oos_mask.sum())
            if oos_mask.any():
                rows_by_h[horizon].append(
                    pd.DataFrame(
                        {
                            "symbol": rec.symbol,
                            "exchange": rec.exchange,
                            "origin_date": idx_dates[oos_mask],
                            "target_end_date": target_end_dates[oos_mask],
                            "realized": y_fwd[oos_mask],
                            "gpu_g3": g3_h[oos_mask],
                            "rolling_mean": roll60_h[oos_mask],
                            "har_rv": har_h[oos_mask],
                            "g0_20d": np.maximum(base20_h[oos_mask], 1e-12),
                        }
                    )
                )
        per_ticker_coverage.append(ticker_cov)

    # First pass: compute raw comparisons for all 3 horizons so we can apply Holm-Bonferroni
    # across the 6 predeclared primary pooled tests (3 horizons x 2 production comparators).
    horizon_results: dict[int, dict[str, Any]] = {}
    primary_p_keys: list[tuple[int, str]] = []
    primary_p_vals: list[float] = []

    for horizon in g3.G3_HORIZONS:
        df_h = pd.concat(rows_by_h[horizon], ignore_index=True)
        df_us = df_h[df_h["exchange"] == "US"].reset_index(drop=True)
        df_lse = df_h[df_h["exchange"] == "LSE"].reset_index(drop=True)

        pooled_vs_roll = _score_volatility_comparator(
            df_h, horizon=horizon, cand_col="gpu_g3", base_col="rolling_mean"
        )
        pooled_vs_har = _score_volatility_comparator(
            df_h, horizon=horizon, cand_col="gpu_g3", base_col="har_rv"
        )
        pooled_vs_g0 = _score_volatility_comparator(
            df_h, horizon=horizon, cand_col="gpu_g3", base_col="g0_20d"
        )
        us_vs_roll = _score_volatility_comparator(
            df_us, horizon=horizon, cand_col="gpu_g3", base_col="rolling_mean"
        )
        lse_vs_roll = _score_volatility_comparator(
            df_lse, horizon=horizon, cand_col="gpu_g3", base_col="rolling_mean"
        )
        us_vs_har = _score_volatility_comparator(
            df_us, horizon=horizon, cand_col="gpu_g3", base_col="har_rv"
        )
        lse_vs_har = _score_volatility_comparator(
            df_lse, horizon=horizon, cand_col="gpu_g3", base_col="har_rv"
        )

        # Annualized variance Mincer-Zarnowitz tests (Driscoll-Kraay HAC lag = H-1 and 2*(H-1))
        scale = TRADING_SESSIONS_PER_YEAR / float(horizon)
        mz_models: dict[str, Any] = {}
        for m_col in ("gpu_g3", "rolling_mean", "har_rv", "g0_20d"):
            mz_primary = mincer_zarnowitz_panel_test(
                df_h["realized"].to_numpy() * scale,
                df_h[m_col].to_numpy() * scale,
                df_h["origin_date"].to_numpy(),
                bandwidth=horizon - 1,
            )
            mz_2x = mincer_zarnowitz_panel_test(
                df_h["realized"].to_numpy() * scale,
                df_h[m_col].to_numpy() * scale,
                df_h["origin_date"].to_numpy(),
                bandwidth=2 * (horizon - 1),
            )
            mz_models[m_col] = {
                **mz_primary,
                "sensitivity_2x_lag": {
                    "bandwidth": mz_2x["bandwidth"],
                    "wald_chi2": mz_2x["wald_chi2"],
                    "p_wald_chi2": mz_2x["p_wald_chi2"],
                    "intercept_rel_ci95": mz_2x["intercept_rel_ci95"],
                    "slope_ci95": mz_2x["slope_ci95"],
                    "mz_gate_pass": mz_2x["mz_gate_pass"],
                },
            }

        # Sub-panel MZ for G3 (US vs LSE)
        mz_g3_us = mincer_zarnowitz_panel_test(
            df_us["realized"].to_numpy() * scale,
            df_us["gpu_g3"].to_numpy() * scale,
            df_us["origin_date"].to_numpy(),
            bandwidth=horizon - 1,
        )
        mz_g3_lse = mincer_zarnowitz_panel_test(
            df_lse["realized"].to_numpy() * scale,
            df_lse["gpu_g3"].to_numpy() * scale,
            df_lse["origin_date"].to_numpy(),
            bandwidth=horizon - 1,
        )

        # Strictly post-freeze sub-window diagnostic (>= 2026-09-08)
        df_pf = df_h[df_h["origin_date"] >= post_freeze_ts].reset_index(drop=True)
        post_freeze_diag: dict[str, Any]
        if len(df_pf) > 0 and df_pf["origin_date"].nunique() >= max(horizon + 2, 5):
            pf_vs_roll = _score_volatility_comparator(
                df_pf, horizon=horizon, cand_col="gpu_g3", base_col="rolling_mean", min_origins=10
            )
            post_freeze_diag = {
                "origin_start": df_pf["origin_date"].min().date().isoformat(),
                "origin_end": df_pf["origin_date"].max().date().isoformat(),
                "common_origins": pf_vs_roll["common_origins"],
                "common_dates": pf_vs_roll["common_dates"],
                "mean_delta_qlike_vs_rolling_60d": pf_vs_roll["mean_delta_qlike"],
                "hac_primary_p_two_sided": pf_vs_roll["hac_primary"]["p_two_sided"],
                "hac_primary_ci95": pf_vs_roll["hac_primary"]["ci95"],
                "asset_win_rate_vs_rolling_60d": pf_vs_roll["asset_win_rate"],
            }
        else:
            post_freeze_diag = {
                "status": "too_few_matured_dates",
                "common_origins": int(len(df_pf)),
            }

        horizon_results[horizon] = {
            "horizon": horizon,
            "origin_start": df_h["origin_date"].min().date().isoformat(),
            "origin_end": df_h["origin_date"].max().date().isoformat(),
            "target_end_max": df_h["target_end_date"].max().date().isoformat(),
            "pooled_vs_rolling_60d": pooled_vs_roll,
            "pooled_vs_har_rv": pooled_vs_har,
            "pooled_vs_g0_20d_diagnostic": pooled_vs_g0,
            "us_vs_rolling_60d": us_vs_roll,
            "lse_vs_rolling_60d": lse_vs_roll,
            "us_vs_har_rv": us_vs_har,
            "lse_vs_har_rv": lse_vs_har,
            "mincer_zarnowitz": mz_models,
            "mincer_zarnowitz_g3_us": mz_g3_us,
            "mincer_zarnowitz_g3_lse": mz_g3_lse,
            "post_freeze_subwindow": post_freeze_diag,
        }
        primary_p_keys.append((horizon, "pooled_vs_rolling_60d"))
        primary_p_vals.append(pooled_vs_roll["hac_primary"]["p_two_sided"])
        primary_p_keys.append((horizon, "pooled_vs_har_rv"))
        primary_p_vals.append(pooled_vs_har["hac_primary"]["p_two_sided"])

    # Step-down Holm-Bonferroni across the 6 primary pooled tests
    holm_adjusted = holm_bonferroni(primary_p_vals)
    for (horizon, key), p_holm in zip(primary_p_keys, holm_adjusted, strict=True):
        horizon_results[horizon][key]["hac_primary"]["p_holm"] = p_holm

    # Also compute Holm-Bonferroni within US (3 tests) and LSE (3 tests) vs rolling_mean
    for sub_key in ("us_vs_rolling_60d", "lse_vs_rolling_60d"):
        raw_ps = [horizon_results[h][sub_key]["hac_primary"]["p_two_sided"] for h in g3.G3_HORIZONS]
        for h, p_adj in zip(g3.G3_HORIZONS, holm_bonferroni(raw_ps), strict=True):
            horizon_results[h][sub_key]["hac_primary"]["p_holm"] = p_adj

    # Evaluate binary PROMOTE_G3_TO_AUTO gate per horizon
    verdicts_per_horizon: dict[str, Any] = {}
    for horizon in g3.G3_HORIZONS:
        res = horizon_results[horizon]
        gate = evaluate_volatility_horizon_gate(
            vs_rolling=res["pooled_vs_rolling_60d"],
            vs_har=res["pooled_vs_har_rv"],
            us_vs_rolling=res["us_vs_rolling_60d"],
            lse_vs_rolling=res["lse_vs_rolling_60d"],
            mz_g3=res["mincer_zarnowitz"]["gpu_g3"],
        )
        res["promotion_gate"] = gate
        verdicts_per_horizon[str(horizon)] = gate["verdict"]

    overall_promote = all(v == "PASS" for v in verdicts_per_horizon.values())
    return {
        "oos_start": oos_start,
        "as_of": as_of,
        "min_common_origins_per_asset": MIN_COMMON_ORIGINS_PER_ASSET,
        "asset_win_rate_threshold": ASSET_WIN_RATE_THRESHOLD,
        "serving_parity_max_rel_error": {str(h): float(err) for h, err in parity_checks.items()},
        "per_ticker_coverage": per_ticker_coverage,
        "horizons": {str(h): horizon_results[h] for h in g3.G3_HORIZONS},
        "verdicts_per_horizon": verdicts_per_horizon,
        "PROMOTE_G3_TO_AUTO": "PASS" if overall_promote else "FAIL",
    }


# ---------------------------------------------------------------------------
# Step 5: Exact Served Ridge Price Predictor Walk-Forward & Conformal Gate
# ---------------------------------------------------------------------------
def _fit_predict_robust_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    *,
    alpha: float = 8.0,
) -> np.ndarray:
    """Closed-form RobustScaler + Ridge(alpha=8.0) matching sklearn to ~1e-15."""
    q25, med, q75 = np.percentile(x_train, [25.0, 50.0, 75.0], axis=0)
    scale = q75 - q25
    scale = np.where(np.abs(scale) <= 1e-15, 1.0, scale)
    xs = (x_train - med) / scale
    xm = xs.mean(axis=0)
    ym = y_train.mean(axis=0)
    xc = xs - xm
    yc = y_train - ym
    gram = xc.T @ xc
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    weights = np.linalg.solve(gram, xc.T @ yc)
    xs_pred = (x_pred - med) / scale
    return (xs_pred - xm) @ weights + ym


def exact_served_ridge_at_origin(
    all_features: pd.DataFrame,
    all_targets: np.ndarray,
    all_feature_positions: np.ndarray,
    origin_pos: int,
) -> dict[str, np.ndarray]:
    """Run exact ``simple_forecast.py`` Ridge selection/fitting/band logic at ``origin_pos``.

    Uses ONLY bars ``0..origin_pos`` (``raw_rows = origin_pos + 1``) so every training
    and validation target satisfies ``target_pos <= origin_pos`` (strict target maturity).
    """
    raw_rows = origin_pos + 1
    if raw_rows < MIN_HISTORY_ROWS:
        raise ValueError("Insufficient history at origin")

    # Labelled rows strictly matured by origin_pos: p + FORECAST_DAYS < raw_rows <=> p + 7 <= origin_pos
    matured_mask = (all_feature_positions + FORECAST_DAYS) < raw_rows
    matured_positions = all_feature_positions[matured_mask]
    x_matured = all_features.iloc[matured_mask].to_numpy(dtype=np.float64)
    y_matured = all_targets[matured_mask]

    # Exact chronological_masks from simple_forecast.py (70/15/15 with 7-session purge)
    ds = ForecastDataset(
        features=all_features,
        targets=y_matured,
        origin_positions=matured_positions,
        labelled_count=len(matured_positions),
    )
    train_mask, val_mask, _test_mask = chronological_masks(ds, raw_rows)

    # 1. Validation residual quantiles (exact simple_forecast.py lines 802-834)
    val_raw = _fit_predict_robust_ridge(
        x_matured[train_mask], y_matured[train_mask], x_matured[val_mask], alpha=8.0
    )
    val_preds = _clip_predictions(val_raw, y_matured[train_mask])
    val_residuals = y_matured[val_mask] - val_preds
    residual_low = np.quantile(val_residuals, 0.10, axis=0)
    residual_high = np.quantile(val_residuals, 0.90, axis=0)

    # 2. Production refit on all matured labelled rows (exact simple_forecast.py lines 867-874)
    latest_row_idx = int(np.searchsorted(all_feature_positions, origin_pos))
    if (
        latest_row_idx >= len(all_feature_positions)
        or all_feature_positions[latest_row_idx] != origin_pos
    ):
        raise ValueError("Latest origin lacks valid price features")
    x_latest = all_features.iloc[[latest_row_idx]].to_numpy(dtype=np.float64)
    prod_raw = _fit_predict_robust_ridge(x_matured, y_matured, x_latest, alpha=8.0)
    forecast_returns = _clip_predictions(prod_raw, y_matured).reshape(-1)

    low_returns = np.minimum(forecast_returns + residual_low, forecast_returns + residual_high)
    high_returns = np.maximum(forecast_returns + residual_low, forecast_returns + residual_high)
    return {
        "forecast_returns": forecast_returns,
        "lower_returns": low_returns,
        "upper_returns": high_returns,
        "residual_low": residual_low,
        "residual_high": residual_high,
    }


def _summarize_coverage_subset(
    sub: pd.DataFrame,
    *,
    low_prefix: str,
    high_prefix: str,
) -> dict[str, Any]:
    """Compute pointwise marginal (pooled & h=1..7), tail miss, width, and simultaneous 7d coverage."""
    n = len(sub)
    if n == 0:
        return {"origins": 0, "pointwise_marginal_coverage": float("nan")}

    in_band_cols = []
    lower_miss_cols = []
    upper_miss_cols = []
    width_cols = []
    per_h: dict[int, dict[str, float]] = {}

    for h in range(1, FORECAST_DAYS + 1):
        y = sub[f"y_h{h}"].to_numpy(dtype=np.float64)
        lo = sub[f"{low_prefix}_h{h}"].to_numpy(dtype=np.float64)
        hi = sub[f"{high_prefix}_h{h}"].to_numpy(dtype=np.float64)
        covered = (y >= lo) & (y <= hi)
        lo_miss = y < lo
        hi_miss = y > hi
        width_pct = 100.0 * (np.exp(hi) - np.exp(lo))
        in_band_cols.append(covered)
        lower_miss_cols.append(lo_miss)
        upper_miss_cols.append(hi_miss)
        width_cols.append(width_pct)
        per_h[h] = {
            "coverage": float(np.mean(covered)),
            "lower_tail_miss": float(np.mean(lo_miss)),
            "upper_tail_miss": float(np.mean(hi_miss)),
            "mean_width_pct_of_origin": float(np.mean(width_pct)),
        }

    in_mat = np.column_stack(in_band_cols)
    lo_mat = np.column_stack(lower_miss_cols)
    hi_mat = np.column_stack(upper_miss_cols)
    w_mat = np.column_stack(width_cols)
    simultaneous_7d = np.all(in_mat, axis=1)

    return {
        "origins": int(n),
        "dates": int(sub["origin_date"].nunique()),
        "pointwise_marginal_coverage": float(np.mean(in_mat)),
        "simultaneous_7d_path_coverage": float(np.mean(simultaneous_7d)),
        "lower_tail_miss": float(np.mean(lo_mat)),
        "upper_tail_miss": float(np.mean(hi_mat)),
        "mean_width_pct_of_origin": float(np.mean(w_mat)),
        "per_horizon": {str(h): per_h[h] for h in range(1, FORECAST_DAYS + 1)},
    }


def _summarize_persistence_mae(sub: pd.DataFrame) -> dict[str, Any]:
    """Scale-independent walk-forward persistence MAE ratio with HAC + circular block bootstrap."""
    y_mat = np.column_stack([sub[f"y_h{h}"].to_numpy(dtype=np.float64) for h in range(1, 8)])
    p_mat = np.column_stack([sub[f"pred_h{h}"].to_numpy(dtype=np.float64) for h in range(1, 8)])

    model_abs_err = np.abs(np.exp(p_mat) - np.exp(y_mat))
    pers_abs_err = np.abs(1.0 - np.exp(y_mat))

    row_model_mae = model_abs_err.mean(axis=1)
    row_pers_mae = pers_abs_err.mean(axis=1)
    row_diff = row_pers_mae - row_model_mae  # positive favors model over persistence

    daily = (
        pd.DataFrame(
            {
                "origin_date": sub["origin_date"].to_numpy(),
                "model_mae": row_model_mae,
                "pers_mae": row_pers_mae,
                "diff": row_diff,
            }
        )
        .groupby("origin_date", sort=True)
        .mean()
    )
    hac_lag6 = bartlett_hac_summary(daily["diff"].to_numpy(), bandwidth=FORECAST_DAYS - 1)
    hac_lag12 = bartlett_hac_summary(daily["diff"].to_numpy(), bandwidth=2 * (FORECAST_DAYS - 1))
    boot_ratio = circular_block_bootstrap_ratio(
        daily["model_mae"].to_numpy(),
        daily["pers_mae"].to_numpy(),
        block_length=2 * FORECAST_DAYS,
        n_boot=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED + 77,
    )

    per_h: dict[str, Any] = {}
    for h_idx in range(FORECAST_DAYS):
        h_model = float(np.mean(model_abs_err[:, h_idx])) * 100.0
        h_pers = float(np.mean(pers_abs_err[:, h_idx])) * 100.0
        h_daily = (
            pd.DataFrame(
                {
                    "origin_date": sub["origin_date"].to_numpy(),
                    "diff": pers_abs_err[:, h_idx] - model_abs_err[:, h_idx],
                }
            )
            .groupby("origin_date", sort=True)["diff"]
            .mean()
        )
        h_hac = bartlett_hac_summary(h_daily.to_numpy(), bandwidth=h_idx)
        per_h[str(h_idx + 1)] = {
            "model_mae_pct": h_model,
            "persistence_mae_pct": h_pers,
            "mae_ratio": float(h_model / h_pers) if h_pers > 0.0 else float("nan"),
            "hac_t_stat_pers_minus_model": h_hac["t_stat"],
            "hac_p_two_sided": h_hac["p_two_sided"],
        }

    overall_model = float(np.mean(model_abs_err)) * 100.0
    overall_pers = float(np.mean(pers_abs_err)) * 100.0
    ratio = float(overall_model / overall_pers) if overall_pers > 0.0 else float("nan")
    beats_persistence = bool(
        ratio < 1.0
        and hac_lag6["mean"] > 0.0
        and hac_lag6["ci95"][0] > 0.0
        and boot_ratio["ci95"][1] < 1.0
    )
    return {
        "origins": int(len(sub)),
        "dates": int(len(daily)),
        "model_mae_pct": overall_model,
        "persistence_mae_pct": overall_pers,
        "mae_ratio": ratio,
        "beats_persistence_significantly": beats_persistence,
        "hac_lag6_pers_minus_model": hac_lag6,
        "hac_lag12_pers_minus_model": hac_lag12,
        "block_bootstrap_ratio": boot_ratio,
        "per_horizon": per_h,
    }


def evaluate_price_walk_forward(
    records: list[TickerDataRecord],
    *,
    oos_start: str = OOS_START_DATE,
    as_of: str,
) -> dict[str, Any]:
    """Walk-forward evaluation of exact served Ridge predictor with strict target maturity."""
    oos_start_ts = pd.Timestamp(oos_start)
    as_of_ts = pd.Timestamp(as_of)

    all_rows: list[dict[str, Any]] = []
    parity_max_diff = 0.0

    for rec in records:
        df = rec.frame
        feats = build_features(df)
        close = df["Close"].to_numpy(dtype=np.float64)
        log_close = np.log(close)
        pos_by_date = pd.Series(np.arange(len(df), dtype=np.int64), index=df.index)
        feat_positions = pos_by_date.loc[feats.index].to_numpy(dtype=np.int64)

        # Precompute realized 1..7d targets (strictly indexed; only used when matured!)
        all_targets = np.full((len(feats), FORECAST_DAYS), np.nan, dtype=np.float64)
        for idx_row, p in enumerate(feat_positions):
            if p + FORECAST_DAYS < len(df):
                for h in range(FORECAST_DAYS):
                    all_targets[idx_row, h] = float(log_close[p + h + 1] - log_close[p])

        # Strictly causal volatility regime at each origin t: vol_20d / vol_60d
        logret = pd.Series(np.log(df["Close"] / df["Close"].shift(1)), index=df.index)
        vol20 = logret.rolling(20, min_periods=20).std(ddof=1)
        vol60 = logret.rolling(60, min_periods=60).std(ddof=1)
        vol_ratio = (vol20 / vol60).to_numpy(dtype=np.float64)

        # Start walk-forward warmup 260 trading days before OOS_START so strict-maturity
        # online conformal has a full trailing matured residual pool on day 1 of OOS!
        oos_indices = np.where(feats.index >= oos_start_ts)[0]
        if len(oos_indices) == 0:
            continue
        first_oos_feat_idx = int(oos_indices[0])
        warmup_start_feat_idx = max(
            0, first_oos_feat_idx - (ONLINE_CONFORMAL_WINDOW + FORECAST_DAYS)
        )

        wf_positions: list[int] = []
        wf_residuals: list[np.ndarray] = []

        # Spot-check parity on the final matured OOS origin against full-slice simple_forecast logic
        last_matured_pos = int(feat_positions[-1]) - FORECAST_DAYS

        for feat_idx in range(warmup_start_feat_idx, len(feats)):
            origin_pos = int(feat_positions[feat_idx])
            if origin_pos + FORECAST_DAYS >= len(df):
                break
            target_end_date = df.index[origin_pos + FORECAST_DAYS]
            if target_end_date > as_of_ts:
                break
            if origin_pos + 1 < MIN_HISTORY_ROWS:
                continue

            try:
                out = exact_served_ridge_at_origin(feats, all_targets, feat_positions, origin_pos)
            except ValueError:
                continue

            y_true = all_targets[feat_idx]
            if not np.isfinite(y_true).all():
                continue

            # Parity spot-check on one concrete slice per ticker: build_features(df.iloc[:origin_pos+1])
            if origin_pos == last_matured_pos:
                sliced_df = df.iloc[: origin_pos + 1]
                sliced_feats = build_features(sliced_df)
                diff_feat = float(
                    np.max(
                        np.abs(
                            sliced_feats.iloc[-1].to_numpy(dtype=np.float64)
                            - feats.iloc[feat_idx].to_numpy(dtype=np.float64)
                        )
                    )
                )
                parity_max_diff = max(parity_max_diff, diff_feat)

            origin_date = df.index[origin_pos]
            if origin_date >= oos_start_ts:
                # Strict target-matured online conformal (pointwise + simultaneous 7d)
                past_pos_arr = np.asarray(wf_positions, dtype=np.int64)
                past_res_arr = (
                    np.vstack(wf_residuals)
                    if wf_residuals
                    else np.empty((0, FORECAST_DAYS), dtype=np.float64)
                )

                online_lo = np.full(FORECAST_DAYS, np.nan, dtype=np.float64)
                online_hi = np.full(FORECAST_DAYS, np.nan, dtype=np.float64)
                for h in range(FORECAST_DAYS):
                    q_res = strict_matured_conformal_quantiles(
                        past_res_arr[:, h],
                        past_pos_arr,
                        origin_pos,
                        horizon_days=h + 1,
                    )
                    if q_res is not None:
                        q_lo, q_hi, _ = q_res
                        online_lo[h] = out["forecast_returns"][h] + q_lo
                        online_hi[h] = out["forecast_returns"][h] + q_hi

                # Simultaneous 7-day max-normalized conformal (strictly s + 7 <= origin_pos)
                sim_lo = np.full(FORECAST_DAYS, np.nan, dtype=np.float64)
                sim_hi = np.full(FORECAST_DAYS, np.nan, dtype=np.float64)
                matured_all7 = (past_pos_arr + FORECAST_DAYS) <= origin_pos
                if int(matured_all7.sum()) >= ONLINE_CONFORMAL_MIN_MATURED:
                    res7 = past_res_arr[matured_all7][-ONLINE_CONFORMAL_WINDOW:]
                    scales = np.maximum(
                        np.quantile(np.abs(res7), 0.50, axis=0),
                        1e-6,
                    )
                    max_scores = np.max(np.abs(res7) / scales[None, :], axis=1)
                    n_m = len(max_scores)
                    q_idx = min(
                        n_m - 1,
                        int(np.ceil((1.0 - ONLINE_CONFORMAL_ALPHA) * (n_m + 1))) - 1,
                    )
                    q_sim = float(np.sort(max_scores)[q_idx])
                    sim_lo = out["forecast_returns"] - q_sim * scales
                    sim_hi = out["forecast_returns"] + q_sim * scales

                vr = float(vol_ratio[origin_pos]) if np.isfinite(vol_ratio[origin_pos]) else 1.0
                regime = "elevated" if vr > ELEVATED_VOL_RATIO_THRESHOLD else "calm"

                row_dict: dict[str, Any] = {
                    "symbol": rec.symbol,
                    "exchange": rec.exchange,
                    "origin_date": origin_date,
                    "target_end_date": target_end_date,
                    "vol_ratio_20_60": vr,
                    "regime": regime,
                }
                for h in range(FORECAST_DAYS):
                    hd = h + 1
                    row_dict[f"y_h{hd}"] = float(y_true[h])
                    row_dict[f"pred_h{hd}"] = float(out["forecast_returns"][h])
                    row_dict[f"served_lo_h{hd}"] = float(out["lower_returns"][h])
                    row_dict[f"served_hi_h{hd}"] = float(out["upper_returns"][h])
                    row_dict[f"online_lo_h{hd}"] = float(online_lo[h])
                    row_dict[f"online_hi_h{hd}"] = float(online_hi[h])
                    row_dict[f"sim_lo_h{hd}"] = float(sim_lo[h])
                    row_dict[f"sim_hi_h{hd}"] = float(sim_hi[h])
                all_rows.append(row_dict)

            # Record residual; note it is ONLY admitted at future origins t >= origin_pos + h!
            wf_positions.append(origin_pos)
            wf_residuals.append(y_true - out["forecast_returns"])

    wf_df = pd.DataFrame(all_rows)
    if wf_df.empty:
        raise RuntimeError("Price walk-forward produced zero out-of-sample rows")

    us_df = wf_df[wf_df["exchange"] == "US"]
    lse_df = wf_df[wf_df["exchange"] == "LSE"]
    calm_df = wf_df[wf_df["regime"] == "calm"]
    elev_df = wf_df[wf_df["regime"] == "elevated"]

    served_pooled = _summarize_coverage_subset(
        wf_df, low_prefix="served_lo", high_prefix="served_hi"
    )
    served_us = _summarize_coverage_subset(us_df, low_prefix="served_lo", high_prefix="served_hi")
    served_lse = _summarize_coverage_subset(lse_df, low_prefix="served_lo", high_prefix="served_hi")
    served_calm = _summarize_coverage_subset(
        calm_df, low_prefix="served_lo", high_prefix="served_hi"
    )
    served_elev = _summarize_coverage_subset(
        elev_df, low_prefix="served_lo", high_prefix="served_hi"
    )
    served_us_calm = _summarize_coverage_subset(
        us_df[us_df["regime"] == "calm"], low_prefix="served_lo", high_prefix="served_hi"
    )
    served_us_elev = _summarize_coverage_subset(
        us_df[us_df["regime"] == "elevated"], low_prefix="served_lo", high_prefix="served_hi"
    )
    served_lse_calm = _summarize_coverage_subset(
        lse_df[lse_df["regime"] == "calm"], low_prefix="served_lo", high_prefix="served_hi"
    )
    served_lse_elev = _summarize_coverage_subset(
        lse_df[lse_df["regime"] == "elevated"], low_prefix="served_lo", high_prefix="served_hi"
    )

    online_valid = wf_df.dropna(subset=[f"online_lo_h{h}" for h in range(1, 8)])
    online_pointwise = {
        "pooled": _summarize_coverage_subset(
            online_valid, low_prefix="online_lo", high_prefix="online_hi"
        ),
        "us": _summarize_coverage_subset(
            online_valid[online_valid["exchange"] == "US"],
            low_prefix="online_lo",
            high_prefix="online_hi",
        ),
        "lse": _summarize_coverage_subset(
            online_valid[online_valid["exchange"] == "LSE"],
            low_prefix="online_lo",
            high_prefix="online_hi",
        ),
        "calm": _summarize_coverage_subset(
            online_valid[online_valid["regime"] == "calm"],
            low_prefix="online_lo",
            high_prefix="online_hi",
        ),
        "elevated": _summarize_coverage_subset(
            online_valid[online_valid["regime"] == "elevated"],
            low_prefix="online_lo",
            high_prefix="online_hi",
        ),
    }

    sim_valid = wf_df.dropna(subset=[f"sim_lo_h{h}" for h in range(1, 8)])
    online_simultaneous = {
        "pooled": _summarize_coverage_subset(sim_valid, low_prefix="sim_lo", high_prefix="sim_hi"),
        "us": _summarize_coverage_subset(
            sim_valid[sim_valid["exchange"] == "US"], low_prefix="sim_lo", high_prefix="sim_hi"
        ),
        "lse": _summarize_coverage_subset(
            sim_valid[sim_valid["exchange"] == "LSE"], low_prefix="sim_lo", high_prefix="sim_hi"
        ),
        "calm": _summarize_coverage_subset(
            sim_valid[sim_valid["regime"] == "calm"], low_prefix="sim_lo", high_prefix="sim_hi"
        ),
        "elevated": _summarize_coverage_subset(
            sim_valid[sim_valid["regime"] == "elevated"],
            low_prefix="sim_lo",
            high_prefix="sim_hi",
        ),
    }

    mae_pooled = _summarize_persistence_mae(wf_df)
    mae_us = _summarize_persistence_mae(us_df)
    mae_lse = _summarize_persistence_mae(lse_df)

    per_h_cov = {
        int(h): float(info["coverage"]) for h, info in served_pooled["per_horizon"].items()
    }
    price_gate = evaluate_price_band_gate(
        pooled_pointwise_coverage=served_pooled["pointwise_marginal_coverage"],
        per_horizon_coverages=per_h_cov,
        us_pointwise_coverage=served_us["pointwise_marginal_coverage"],
        lse_pointwise_coverage=served_lse["pointwise_marginal_coverage"],
        calm_pointwise_coverage=served_calm["pointwise_marginal_coverage"],
        elevated_pointwise_coverage=served_elev["pointwise_marginal_coverage"],
        pooled_lower_tail_miss=served_pooled["lower_tail_miss"],
        pooled_upper_tail_miss=served_pooled["upper_tail_miss"],
    )

    return {
        "oos_start": oos_start,
        "as_of": as_of,
        "origin_start": wf_df["origin_date"].min().date().isoformat(),
        "origin_end": wf_df["origin_date"].max().date().isoformat(),
        "target_end_max": wf_df["target_end_date"].max().date().isoformat(),
        "feature_causality_parity_max_diff": float(parity_max_diff),
        "regime_definition": f"elevated iff trailing vol_20d / vol_60d > {ELEVATED_VOL_RATIO_THRESHOLD}",
        "served_ridge_band": {
            "pooled": served_pooled,
            "us": served_us,
            "lse": served_lse,
            "calm": served_calm,
            "elevated": served_elev,
            "us_calm": served_us_calm,
            "us_elevated": served_us_elev,
            "lse_calm": served_lse_calm,
            "lse_elevated": served_lse_elev,
        },
        "diagnostic_matured_online_pointwise_conformal": online_pointwise,
        "diagnostic_matured_online_simultaneous_7d_conformal": online_simultaneous,
        "persistence_mae_ratio": {
            "pooled": mae_pooled,
            "us": mae_us,
            "lse": mae_lse,
        },
        "calibration_gate": price_gate,
        "UPGRADE_PRICE_BAND_LABEL_TO_CALIBRATED": price_gate["verdict"],
    }


# ---------------------------------------------------------------------------
# End-to-End Runner & CLI
# ---------------------------------------------------------------------------
def run_full_validation(
    *,
    as_of: str | None = None,
    cache_dir: Path | None = None,
    refresh: bool = False,
    tickers: tuple[str, ...] = TRI_EXCHANGE_TICKERS,
    years: int = 8,
) -> dict[str, Any]:
    """Execute the complete Milestone 05 empirical validation suite."""
    t0 = time.perf_counter()
    artifacts = verify_locked_g3_artifacts()
    records, fetch_attrition, resolved_as_of = load_or_fetch_panel(
        tickers=tickers,
        as_of=as_of,
        years=years,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    if not records:
        raise RuntimeError("No tickers available for validation")

    msft_records = [r for r in records if r.symbol == "MSFT"]
    print(
        f"[m5] Loaded {len(records)}/{len(tickers)} tickers as_of={resolved_as_of}; running MSFT trace...",
        file=sys.stderr,
        flush=True,
    )
    msft_trace = trace_msft_snapshot(msft_records[0]) if msft_records else None

    print(
        "[m5] Evaluating volatility promotion gate (G3 vs rolling_mean & har_rv)...",
        file=sys.stderr,
        flush=True,
    )
    vol_report = evaluate_volatility_panel(records, oos_start=OOS_START_DATE, as_of=resolved_as_of)
    print(
        f"[m5] Volatility done in {time.perf_counter() - t0:.1f}s (PROMOTE_G3_TO_AUTO={vol_report['PROMOTE_G3_TO_AUTO']}); "
        "running walk-forward price calibration gate...",
        file=sys.stderr,
        flush=True,
    )
    price_report = evaluate_price_walk_forward(
        records, oos_start=OOS_START_DATE, as_of=resolved_as_of
    )
    print(
        f"[m5] Price calibration done in {time.perf_counter() - t0:.1f}s "
        f"(UPGRADE_PRICE_BAND_LABEL_TO_CALIBRATED={price_report['UPGRADE_PRICE_BAND_LABEL_TO_CALIBRATED']})",
        file=sys.stderr,
        flush=True,
    )

    elapsed = time.perf_counter() - t0
    return {
        "schema_version": "m5-validation-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": float(elapsed),
        "protocol": {
            "panel_name": "TRI_EXCHANGE_TICKERS",
            "panel_requested_count": len(tickers),
            "panel_loaded_count": len(records),
            "us_tickers": [r.symbol for r in records if r.exchange == "US"],
            "lse_tickers": [r.symbol for r in records if r.exchange == "LSE"],
            "as_of": resolved_as_of,
            "oos_start": OOS_START_DATE,
            "post_freeze_start": POST_FREEZE_START_DATE,
            "min_common_origins_per_asset": MIN_COMMON_ORIGINS_PER_ASSET,
            "asset_win_rate_threshold": ASSET_WIN_RATE_THRESHOLD,
            "significance_alpha": SIGNIFICANCE_ALPHA,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "mz_intercept_rel_bound": MZ_INTERCEPT_REL_BOUND,
            "mz_slope_bounds": list(MZ_SLOPE_BOUNDS),
            "mz_mean_ratio_bounds": list(MZ_MEAN_RATIO_BOUNDS),
            "price_coverage_bounds": list(PRICE_COVERAGE_BOUNDS),
            "price_tail_miss_bounds": list(PRICE_TAIL_MISS_BOUNDS),
        },
        "artifacts": artifacts,
        "fetch_attrition": fetch_attrition,
        "ticker_audit": [
            {
                "symbol": r.symbol,
                "exchange": r.exchange,
                "currency": r.currency,
                "rows": r.rows,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "m2_repairs_count": len(r.repairs),
                "m2_repairs": list(r.repairs),
                "raw_source_fingerprint": r.raw_source_fingerprint,
                "data_fingerprint": r.data_fingerprint,
            }
            for r in records
        ],
        "msft_trace": msft_trace,
        "volatility_validation": vol_report,
        "price_validation": price_report,
        "final_verdicts": {
            "PROMOTE_G3_TO_AUTO": vol_report["PROMOTE_G3_TO_AUTO"],
            "PROMOTE_G3_TO_AUTO_BY_HORIZON": vol_report["verdicts_per_horizon"],
            "UPGRADE_PRICE_BAND_LABEL_TO_CALIBRATED": price_report[
                "UPGRADE_PRICE_BAND_LABEL_TO_CALIBRATED"
            ],
            "PRICE_RIDGE_BEATS_PERSISTENCE_MAE": price_report["persistence_mae_ratio"]["pooled"][
                "beats_persistence_significantly"
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of",
        default=None,
        help="Common completed session cutoff YYYY-MM-DD (default: min latest US/LSE completed session)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory to cache normalized OHLCV parquet files for deterministic reruns",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force fresh Yahoo downloads even if cache-dir already has data",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write full machine-readable JSON results",
    )
    args = parser.parse_args()

    result = run_full_validation(
        as_of=args.as_of,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result["final_verdicts"], indent=2))
    print(f"Elapsed: {result['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
