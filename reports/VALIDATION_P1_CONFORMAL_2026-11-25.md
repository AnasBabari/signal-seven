# P1 Price Uncertainty Band Conformal Calibration Report

- **Date:** `2026-11-25`
- **Protocol:** Strictly target-matured walk-forward evaluation (`origin_pos + h <= t`, zero future leakage) across **`20,839` out-of-sample origins** on the predeclared 30-ticker tri-exchange M5 panel (`20` US + `10` UK LSE `.L`, `origin_date >= 2023-11-16`)
- **Production Method:** Volatility-scaled online split conformal (`blend_ewma94_v60`: $\hat{\sigma}_t = \sqrt{0.65\,\text{EWMA}_{0.94}(r_t^2) + 0.35\,\text{Var}_{60\text{d}}(r_t)}$, trailing `W = 250` matured residuals, `refit_every = 20`, nominal `80.0%` pointwise coverage, `10%` each tail)

---

## Executive Verdict

- **`UPGRADE_PRICE_BAND_LABEL_TO_CALIBRATED`: `PASS` (`5 / 5` predeclared M5 clauses cleared)**
- **Cold Latency (2,000-row MSFT):** **`41.2 ms`** (closed-form `RobustScaler + Ridge(alpha=8.0)` across 12 causal calibration blocks)

---

## 1. Volatility Normalizer Ablation Across 20,839 OOS Origins

Predeclared M5 gate bounds: Pooled, US, LSE, Calm, Elevated, and every horizon $h \in \{1\dots 7\}$ pointwise marginal coverage $\in [75.0\%, 85.0\%]$; lower and upper tail misses $\in [5.0\%, 15.0\%]$.

| Calibration Variant | M5 Gate Verdict | Pooled | US (20) | LSE (10) | Calm ($\le 1.15$) | Elevated ($> 1.15$) | LSE × Elevated | Lower / Upper Tail Miss | Mean Width (% Origin) | Sim. 7d Full-Path | Per-Horizon `h = 1..7` (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *Legacy Static Validation Split (M5 baseline)* | **`FAIL`** *(3/5 failed)* | `75.87%` | `76.86%` | **`73.90%`** | `76.87%` | **`72.71%`** | **`70.45%`** | `11.71% / 12.42%` | `8.01%` | `45.65%` | `76.96..74.80` (`h=7` FAIL) |
| `unit` (unscaled online conformal) | `PASS` | `77.70%` | `77.83%` | `77.44%` | `78.45%` | `75.38%` | `74.24%` | `10.99% / 11.31%` | `8.04%` | `46.99%` | `78.35, 77.86, 78.03, 77.55, 77.59, 77.42, 77.10` |
| `ewma94` (pure EWMA $\lambda=0.94$) | `FAIL` *(Elev >85%)* | `79.19%` | `79.08%` | `79.41%` | `77.21%` | **`85.26%`** | `83.83%` | `10.24% / 10.57%` | `8.77%` | `50.65%` | `79.83..78.49` |
| `ewma90` (pure EWMA $\lambda=0.90$) | `FAIL` *(Elev >85%)* | `79.34%` | `79.27%` | `79.49%` | `76.72%` | **`87.42%`** | `85.85%` | `10.16% / 10.49%` | `8.99%` | `51.29%` | `80.03..78.59` |
| `vol20` (20d rolling std) | `FAIL` *(Elev >85%)* | `79.34%` | `79.33%` | `79.34%` | `76.15%` | **`89.11%`** | `87.48%` | `10.12% / 10.55%` | `9.13%` | `51.58%` | `79.97..78.69` |
| **`blend_ewma94_v60` (SHIPPED)** | **`PASS` (`5/5`)** | **`79.15%`** | **`79.02%`** | **`79.41%`** | **`77.82%`** | **`83.26%`** | **`81.93%`** | **`10.22% / 10.62%`** | **`8.69%`** | **`50.45%`** | **`79.77, 79.49, 79.51, 79.21, 78.80, 78.63, 78.67`** |

---

## 2. Production Serving & Honest UI Disclosure

1. **Backend (`backend/services/simple_forecast.py`):** Served Ridge forecasts now compute `interval_calibration` via `_compute_conformal_band_offsets(..., vol_scale="blend_ewma94_v60")`, emitting `m5_gate_verdict: "PASS"` alongside exact empirical coverage metadata. Explicit `model=random_forest` and short-history fallbacks continue emitting `m5_gate_verdict: "UNVERIFIED_FALLBACK"`.
2. **Frontend (`frontend/src/utils/estimateAdapter.js`, `PriceChart.jsx`):** Data-driven label upgrade — displays **`80% pointwise conformal range`** when `interval_calibration.m5_gate_verdict === "PASS"` (with inline `ℹ` disclosure transparently noting both the `79.2%` pointwise OOS coverage and `~50%` simultaneous 7-day full-path coverage), and falls back to **`Historical error range (experimental)`** whenever calibration is unverified.
