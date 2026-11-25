# P1 Volatility ML Validation Report: Causal Duan-Smeared HAR-RV & Scale-Free 60d-Anchored G3 v2

- **Date:** `2026-11-25`
- **Untouched Out-of-Sample Reserve:** `origin_date >= 2023-11-16` across the predeclared 30-ticker tri-exchange M5 panel (`20` US + `10` UK LSE `.L`)
- **G3 v2 Training Panel:** `286` stocks (`642,130` rows), strictly trained on `target_end_date < 2022-05-09` with early stopping tuned solely on `2022-05-09..2023-11-15` (zero reserve peeking)

---

## Executive Verdicts

| Candidate | Scope | Promotion Verdict | Binding Empirical Basis |
| :--- | :---: | :---: | :--- |
| **Causal Duan-Smeared `har_rv` (`v5_duan_har`)** | **`H = 5, 10, 20`** *(keep `garch_11` at `H=1`)* | **`PROMOTE` (`PASS`)** | Clears all 5 predeclared M5 pooled clauses at **every horizon**: `+9.0% / +12.5% / +19.3%` QLIKE vs 60d `rolling_mean` (`p_holm = 1.12e-4 / 1.11e-3 / 2.86e-3`), **`93% / 97% / 93%`** asset win rates (**`10/10 = 100%` on LSE** at all 3 horizons!), **Pooled & US Mincer–Zarnowitz PASS** (`W = 0.73..3.02, p = 0.221..0.694, b = 0.960..0.975`), **zero LSE level bias** (`f/y = 0.999..1.029`), `<2ms` runtime, zero ONNX artifacts. |
| **Scale-Free 60d-Anchored `G3 v2` (GPU XGBoost)** | **`H = 1, 5, 10, 20`** | **`HOLD` (`FAIL` MZ)** | Scale-free 24-feature design (`max_scale_diff = 5.76e-8`) halved LSE level bias vs G3 v1 (`1.21–1.35` → `1.09–1.12`), lifted `H=10/20` win rates to **`30/30 (100%)`**, and made `H=20` HAC significant (`t=+3.70, p=2.17e-4`). However, it **still fails Pooled & US Mincer–Zarnowitz calibration** at `H=5/10/20` (`W = 7.77..24.25, p <= 0.021, b = 0.724..0.778`) and is **beaten or tied by zero-artifact Duan-smeared `har_rv`** at `H=10/20`. |

---

## 1. Causal Duan-Smeared `har_rv` (`smearing_exp_cap = 50.0`) vs Legacy `har_rv` vs `rolling_mean` (60d)

Fixing `causal_log_har_forecasts` in `backend/services/volatility_snapshot.py` to apply strictly in-sample causal Duan (1983) smearing ($\hat{s} = \operatorname{clip}(\frac{1}{n}\sum \min(\exp(e_i), 50), 1, 8)$) eliminates the `5.5×–7.8×` geometric-mean downward bias and delivers dramatic, uniform gains across both US and LSE equities:

| Horizon | Legacy `har_rv` QLIKE (`f/y`) | `rolling_mean` 60d QLIKE | **Duan `har_rv` QLIKE** | **ΔQLIKE vs 60d (`Rel. %`)** | **Bartlett HAC `t` (`p_holm`)** | **Asset Wins (Pool / US / LSE)** | **Pooled MZ (`a_rel`, `b`, `f/y`, `W`, `p`)** | **US MZ Pass?** | **LSE `f/y`** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`H = 1`** | `4.4730` (`0.192`) | `1.7759` *(GARCH: `1.7293`)* | **`1.7220`** | `+0.0539 (+3.0%)` | `+4.61` (`1.59e-05`) | `77% / 75% / 80%` | `+0.049, 1.018, 0.934, W=5.37, p=0.068` (`PASS`) | `p=0.044` | `0.984` |
| **`H = 5`** | `3.9527` (`0.169`) | `0.5593` | **`0.5091`** | **`+0.0502 (+9.0%)`** | **`+4.12` (`1.12e-04`)** | **`28/30 (93%) / 90% / 100%`** | **`+0.062, 0.975, 0.962, W=3.02, p=0.221` (`PASS`)** | **`PASS` (`p=0.171`)** | **`0.999`** |
| **`H = 10`** | `4.7218` (`0.145`) | `0.3942` | **`0.3449`** | **`+0.0492 (+12.5%)`** | **`+3.45` (`1.11e-03`)** | **`29/30 (97%) / 95% / 100%`** | **`+0.060, 0.960, 0.979, W=2.26, p=0.324` (`PASS`)** | **`PASS` (`p=0.312`)** | **`1.012`** |
| **`H = 20`** | `6.2960` (`0.116`) | `0.2912` | **`0.2349`** | **`+0.0563 (+19.3%)`** | **`+2.98` (`2.86e-03`)** | **`28/30 (93%) / 90% / 100%`** | **`+0.040, 0.965, 0.994, W=0.73, p=0.694` (`PASS`)** | **`PASS` (`p=0.674`)** | **`1.029`** |

---

## 2. Scale-Free, 60d-Anchored `G3 v2` (CUDA GPU XGBoost) Comparison

| Horizon | Best Val Config (`rounds`) | `G3 v2` QLIKE | ΔQLIKE vs `roll60` (`t`, `p`) | Wins (Pool / US / LSE) | ΔQLIKE vs Duan `har_rv` (`t`) | Pooled MZ (`a_rel`, `b`, `f/y`, `W`, `p`) | LSE `f/y` (v1 was `1.21–1.35`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`H = 1`** | `d3_l20_mcw100` (`301`) | `1.7025` | `+0.0734 (+4.1%, t=+4.46, p=8.1e-6)` *(vs GARCH: `+0.0268, p=3.6e-3`)* | `83% / 90% / 70%` | `+0.0116` (`t=+1.04`, NS) | `+0.062, 0.958, 0.979, W=0.88, p=0.643` (`PASS`) | `1.098` *(LSE MZ `W=49.1` FAIL)* |
| **`H = 5`** | `d3_l20_mcw100` (`598`) | `0.4989` | `+0.0604 (+10.8%, t=+4.14, p=3.4e-5)` | `80% / 90% / 60%` | `+0.0042` (`t=+0.54`, NS) | `+0.215, 0.778, 1.008, W=7.77, p=0.021` (**`FAIL`**) | `1.125` |
| **`H = 10`** | `d3_l5_mcw50` (`40`) | `0.3423` | `+0.0519 (+13.2%, t=+4.41, p=1.0e-5)` | **`100% / 100% / 100%`** | **`-0.0013`** (`t=-0.21`) | `+0.184, 0.773, 1.056, W=11.98, p=0.0025` (**`FAIL`**) | `1.122` |
| **`H = 20`** | `d3_l5_mcw50` (`10`) | `0.2525` | `+0.0387 (+13.3%, t=+3.70, p=2.2e-4)` | **`100% / 100% / 100%`** | **`-0.0179`** (`t=-1.25`) | `+0.237, 0.724, 1.055, W=24.25, p=5.4e-6` (**`FAIL`**) | `1.088` |

## 3. Production Policy Decision

1. **Promote `v5_duan_har` (`H=1: garch_11, H=5/10/20: har_rv`)** as `DEFAULT_VOLATILITY_POLICY` in `backend/services/volatility_contract.py` and `backend/config.py`.
2. **Preserve instant rollback & V1 continuity:** `VOLATILITY_DEFAULT_POLICY=v3_baseline` switches back to 60d `rolling_mean` instantaneously without code changes, and `LIVE_MODEL_POLICY_V1` remains strictly pinned to `rolling_mean` for longitudinal V1 live ledger continuity.
3. **Keep `G3 v2` in research (`research/volatility_structure/g3_v2.py`):** Per-asset Duan-smeared `har_rv` matches or outperforms `G3 v2` with strictly superior calibration (`W = 0.73..3.02` vs `7.77..24.25`), zero cross-market bias, zero ONNX footprint, and `<2ms` execution.
