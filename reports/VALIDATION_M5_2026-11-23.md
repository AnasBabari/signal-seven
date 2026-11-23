# Milestone 05 Empirical ML Validation, Calibration Gate & Promotion Report

- **Report Date:** `2026-11-23`
- **Common Completed Session Cutoff (`as_of`):** `2026-11-20`
- **Strictly Purged Out-of-Sample Window:** `2023-11-16` (`reserve_start`) through `2026-11-20` (right-edge target maturity purged so `target_end_date <= 2026-11-20`)
- **Validation Panel:** Predeclared 30-ticker tri-exchange core panel (`TRI_EXCHANGE_TICKERS`: 20 US = 10 NASDAQ + 10 NYSE; 10 UK LSE `.L`), normalized via Milestone 02 auditable single-pass normalization (`0%` attrition, `30/30` eligible)
- **Reproducible Script:** `research/validation/validate_m5_models.py`
- **Unit Test Suite:** `research/tests/test_m5_validation_gates.py` (`10/10` passing)

---

## Executive Summary & Unambiguous Binary Gate Verdicts

| Gate Identifier | Scope / Horizon | Binary Verdict | Binding Empirical Basis |
| :--- | :--- | :---: | :--- |
| **`PROMOTE_G3_TO_AUTO`** | **Overall Policy (`H = 5, 10, 20`)** | **`FAIL`** | Only `H=5` clears the 5-clause pooled gate; `H=10` fails joint Mincer–Zarnowitz calibration (`W = 8.95, p = 0.0114`) and `H=20` fails both Bartlett HAC QLIKE significance vs 60d `rolling_mean` (`t = +1.82, p_holm = 0.0686`) and LSE asset breadth (`60.0% < 70.0%`). Furthermore, on the LSE sub-panel, `gpu_g3` exhibits severe positive level bias (`+21.0%` to `+34.5%` variance overprediction) and MZ slope collapse (`0.240–0.342`, `p <= 4.39e-19`) at all three horizons. |
| **`PROMOTE_G3_TO_AUTO (H=1)`** | **1-Session (`g3_h1.onnx`)** | **`FAIL` (N/A)** | `backend/volatility_models/g3_h1.onnx` does not exist; retain `garch_11` for `H=1`. |
| **`PROMOTE_G3_TO_AUTO (H=5)`** | **5-Session (`g3_h5.onnx`)** | **`PASS`*** | Clears all 5 predeclared pooled clauses (`ΔQLIKE = +0.0592`, `+10.7%`, `HAC(4) t = +3.64, p_holm = 8.27e-04`, `Boot 95% CI = [+0.0261, +0.0997]`, win rate `24/30 = 80.0%` pooled / `85.0%` US / `70.0%` LSE, pooled MZ `W = 4.73, p = 0.0941`). ***Critical deployment caveat:** pooled MZ passes at the boundary (`a_rel = +0.147 <= 0.150`) only because US underprediction (`meanRatio = 0.945`) and LSE overprediction (`meanRatio = 1.210, W = 84.54, p = 4.39e-19`) offset in the pooled average. Keep `rolling_mean` as the global `auto` default (`VOLATILITY_DEFAULT_POLICY=v3_baseline`). |
| **`PROMOTE_G3_TO_AUTO (H=10)`** | **10-Session (`g3_h10.onnx`)** | **`FAIL`** | Fails `mincer_zarnowitz_joint_calibration`: pooled relative intercept `a_rel = +0.188 > 0.150`, slope `b = 0.796 < 0.800`, joint Wald `W = 8.95 (p = 0.0114)` (`2x-lag W = 7.90, p = 0.0193`); LSE sub-panel QLIKE HAC 95% CI crosses zero (`[-0.0026, +0.0673]`, `p = 0.0696`) and LSE MZ rejects with `W = 203.71 (p = 5.83e-45)`. |
| **`PROMOTE_G3_TO_AUTO (H=20)`** | **20-Session (`g3_h20.onnx`)** | **`FAIL`** | Fails `qlike_beats_rolling_60d_pooled_holm` (`ΔQLIKE = +0.0418`, `HAC(19) t = +1.82, p = 0.0686, p_holm = 0.0686`, `95% CI = [-0.0032, +0.0872]`, `HAC(38) CI = [-0.0090, +0.0930]`, `Boot CI = [-0.0016, +0.0994]`) and fails `win_rate_ge_70pct_vs_rolling_pooled_us_lse` (LSE win rate = `6/10 = 60.0% < 70.0%`). |
| **`UPGRADE_PRICE_BAND_LABEL_TO_CALIBRATED`** | **Served Ridge Uncertainty Band (`simple_forecast.py`)** | **`FAIL`** | Fails 3 of 5 predeclared clauses across `22,535` strictly target-matured walk-forward origins: (1) `h=7` pointwise coverage drops to `74.80% < 75.0%`; (2) LSE pointwise coverage is `73.90% < 75.0%`; (3) Elevated-volatility regime coverage collapses to `72.71% < 75.0%` (`70.45%` on LSE elevated). Simultaneous 7-day path coverage is only **`45.65%`** (`35.24%` on LSE elevated). Retain M4's label **`Historical error range (experimental)`**. |
| **`PRICE_RIDGE_BEATS_PERSISTENCE_MAE`** | **Served Ridge Point Forecast vs Persistence** | **`FAIL` (`False`)** | Walk-forward scale-independent `model_MAE / persistence_MAE = 1.0171` (`2.6891%` vs `2.6440%`), circular block-bootstrap 95% CI `[1.0120, 1.0222]`, date-aggregated Bartlett `HAC(6) t = -6.24 (p = 4.36e-10)`. Ridge underperforms persistence on US (`1.0134`), LSE (`1.0263`), and every single horizon `h = 1..7`. |

### Reproducibility Command

```powershell
$env:PYTHONPATH='backend;research;.'
backend\.venv\Scripts\python.exe research\validation\validate_m5_models.py --as-of 2026-11-20
```

---

## 1. Packaged Artifact Provenance & Integrity Verification

All packaged ONNX artifacts live in `backend/volatility_models/` (`services.g3_volatility._model_dir()`) and trace back to `artifacts/g3_panel_v1/manifest.json` (`created_at = 2026-09-07T15:34:55.900076+00:00`, `freeze_manifest_sha256 = b36e7c9992ce23b29a745de753c4097622410d15eec0baddb5658454b75e574c`).

| Horizon | Artifact Path | Size (Bytes) | Verified `onnx_sha256` | Source `booster_sha256` | Serving Contract | `graph_base_score` | Train Rows (`< 2022-05-09`) | Validation Start | Reserve (OOS) Start | Serving Parity Max Rel Error |
| :---: | :--- | :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`H=1`** | `backend/volatility_models/g3_h1.onnx` | *Absent* | *Does not exist (`g3_h1_onnx_exists = false`)* | *N/A (`garch_11`)* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* |
| **`H=5`** | `backend/volatility_models/g3_h5.onnx` | `242,994` | `8bccde12efb4bf5089ce3cd704cc027653f916f0547ddb5a4404587513d76a93` | `85f1cda89abec121beb2501bce8050b16db78de006bcffb2bd694bbd34e34aa9` | `g3-qlike-base-margin-v2` | `0.5` | `278,765` | `2022-05-09` | `2023-11-16` | `3.58e-07` |
| **`H=10`** | `backend/volatility_models/g3_h10.onnx` | `243,059` | `5014ff1bf2c31e19e00061099748bdba75ca8f4451b0602d7f3384a044f89fbc` | `abf720cf0f8d70a3eb51534d8568c69cb3847236c2626ff31307831aed681e9d` | `g3-qlike-base-margin-v2` | `0.5` | `277,355` | `2022-05-09` | `2023-11-16` | `8.34e-07` |
| **`H=20`** | `backend/volatility_models/g3_h20.onnx` | `243,062` | `79ee8efcf10322ed6fd70f6a382a9aec430bce4eef7e410273b1a30d54665325` | `583f0367b3410cd9e2bfc3facfa79b9dfadf24368b15f2e536d3e73fda3a4148` | `g3-qlike-base-margin-v2` | `0.5` | `274,539` | `2022-05-09` | `2023-11-16` | `7.15e-07` |

### Why `artifacts/gpu_rolling_origin_v1/test_report.json` Is Invalid for Production Promotion

1. **Different Model Weights:** `test_report.json` refit a fresh XGBoost model on all pre-reserve rows (`origin_date < 2023-11-16`, absorbing the `2022-05-09..2023-11-15` validation block into training), whereas the actual packaged `g3_h{5,10,20}.onnx` models shipped in `backend/volatility_models/` were trained exclusively on `origin_date < 2022-05-09` (`target_end_date < 2022-05-09`).
2. **Wrong Baseline Comparator:** `test_report.json` compared G3 only against 20-day `G0` (`persistence`, `H * var_20d`), whereas production `AUTO_MODEL_POLICY` serves 60-day `rolling_mean` (`H * var_60d`). As Section 4 demonstrates, 60-day `rolling_mean` already captures **58%–71%** of G3's apparent QLIKE edge over 20-day `G0`.
3. **Missing Calibration & Sub-Panel Gates:** `test_report.json` did not evaluate joint Mincer–Zarnowitz calibration `(intercept, slope) = (0, 1)` or test UK LSE `.L` equities separately from US equities.

---

## 2. Panel Composition, M2 Normalization Audit & Attrition Report

We evaluate the canonical 30-ticker tri-exchange universe predeclared in both `backend/services/simple_forecast.py` (`TICKER_METADATA`) and `research/volatility_structure/data_utils.py` (`TRI_EXCHANGE_TICKERS`):
- **US Panel (20 tickers, USD):** `AAPL`, `MSFT`, `NVDA`, `GOOGL`, `AMZN`, `META`, `TSLA`, `AMD`, `COST`, `QCOM` (NASDAQ-10); `JPM`, `XOM`, `WMT`, `JNJ`, `CAT`, `KO`, `NEE`, `DIS`, `BAC`, `GE` (NYSE-10).
- **LSE Panel (10 tickers, GBp):** `SHEL.L`, `AZN.L`, `HSBA.L`, `BP.L`, `ULVR.L`, `GSK.L`, `RIO.L`, `BATS.L`, `BARC.L`, `DGE.L`.
- **Attrition Summary:** **0% attrition (`30/30` loaded and eligible)**. Every ticker easily exceeds the predeclared `MIN_COMMON_ORIGINS_PER_ASSET = 250` threshold (`736–758` common OOS origins per ticker across `2023-11-16..2026-11-20`). Milestone 02's single-pass `normalize_daily_bars_with_audit` cleanly repaired the notorious Yahoo `2024-05-01` LSE corrupt OHLC bar (`High < max(O,C)` / `Low > min(O,C)` within `<= 5%` relative drift) across `9/10` LSE tickers without a single rejection.

| Symbol | Exchange | Quote Unit | Total Rows | History Span | M2 Bar Repairs | Common OOS Origins (`H=5 / 10 / 20`) | Mean `ΔQLIKE` vs 60d `rolling_mean` (`H=5`) | Mean `ΔQLIKE` vs 60d `rolling_mean` (`H=10`) | Mean `ΔQLIKE` vs 60d `rolling_mean` (`H=20`) | Normalized `data_fingerprint` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `AAPL` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `+0.1056` | `+0.1091` | `+0.1293` | `89f2aa8256a5` |
| `MSFT` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `+0.1469` | `+0.1162` | `+0.0534` | `94526b638d54` |
| `NVDA` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `+0.0505` | `+0.0575` | `+0.0778` | `2418491e571e` |
| `GOOGL` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `+0.0914` | `+0.0737` | `+0.0560` | `0024aa2eafa7` |
| `AMZN` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `+0.1324` | `+0.1222` | `+0.1186` | `718292118aa7` |
| `META` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `+0.1592` | `+0.1132` | `+0.0945` | `ca040be56813` |
| `TSLA` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `-0.0692` | `-0.0289` | `-0.0283` | `0570918af57a` |
| `AMD` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `-0.0173` | `-0.0474` | `-0.0667` | `2a898ca2b325` |
| `COST` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `+0.0126` | `+0.0188` | `+0.0099` | `f4021103719e` |
| `QCOM` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 1 | `751 / 746 / 736` | `+0.1202` | `+0.0920` | `+0.1382` | `4bc91292a039` |
| `JPM` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `+0.1175` | `+0.1117` | `+0.0891` | `2968630d2ac3` |
| `XOM` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `+0.0060` | `+0.0042` | `-0.0034` | `0b0bef26a52f` |
| `WMT` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `+0.1064` | `+0.0863` | `+0.0581` | `ce5b2f67480b` |
| `JNJ` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `+0.0333` | `+0.0267` | `-0.0270` | `0e382d3fc9a3` |
| `CAT` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `+0.0925` | `+0.0725` | `+0.0577` | `f6c7ff124043` |
| `KO` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `-0.0255` | `-0.0440` | `-0.1304` | `fab9a788a23e` |
| `NEE` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `+0.0091` | `+0.0164` | `+0.0162` | `d798a20694db` |
| `DIS` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 1 | `751 / 746 / 736` | `+0.1807` | `+0.1950` | `+0.1924` | `765f779d9b18` |
| `BAC` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `+0.0720` | `+0.0571` | `+0.0596` | `f6ab6112fb75` |
| `GE` | US | `USD` | 2009 | `2018-11-23..2026-11-20` | 0 | `751 / 746 / 736` | `+0.0812` | `+0.0834` | `+0.1082` | `2492dd782657` |
| `SHEL.L` | LSE | `GBp` | 2019 | `2018-11-23..2026-11-20` | 1 | `758 / 753 / 743` | `-0.0174` | `-0.0215` | `-0.0438` | `bba536579921` |
| `AZN.L` | LSE | `GBp` | 2019 | `2018-11-23..2026-11-20` | 1 | `758 / 753 / 743` | `+0.0988` | `+0.0818` | `+0.0491` | `daa9ed806b19` |
| `HSBA.L` | LSE | `GBp` | 2018 | `2018-11-23..2026-11-20` | 1 | `757 / 752 / 742` | `+0.1005` | `+0.1223` | `+0.1715` | `201fe794cb4a` |
| `BP.L` | LSE | `GBp` | 2018 | `2018-11-23..2026-11-20` | 1 | `757 / 752 / 742` | `-0.0121` | `-0.0161` | `-0.0047` | `054b9da96662` |
| `ULVR.L` | LSE | `GBp` | 2019 | `2018-11-23..2026-11-20` | 1 | `758 / 753 / 743` | `+0.0828` | `+0.0515` | `-0.0072` | `0a8c40961b9d` |
| `GSK.L` | LSE | `GBp` | 2019 | `2018-11-23..2026-11-20` | 0 | `758 / 753 / 743` | `+0.0310` | `+0.0175` | `+0.0089` | `aaa01a9efaa2` |
| `RIO.L` | LSE | `GBp` | 2018 | `2018-11-23..2026-11-20` | 3 | `757 / 752 / 742` | `-0.0222` | `-0.0228` | `-0.0213` | `7af2824ff4c7` |
| `BATS.L` | LSE | `GBp` | 2019 | `2018-11-23..2026-11-20` | 1 | `758 / 753 / 743` | `+0.0123` | `+0.0247` | `+0.0323` | `49b4a6c1adab` |
| `BARC.L` | LSE | `GBp` | 2018 | `2018-11-23..2026-11-20` | 5 | `757 / 752 / 742` | `+0.0711` | `+0.0640` | `+0.0648` | `614104ab37d0` |
| `DGE.L` | LSE | `GBp` | 2018 | `2018-11-23..2026-11-20` | 1 | `757 / 752 / 742` | `+0.0285` | `+0.0217` | `+0.0031` | `4463c869ba1e` |

---

## 3. Concrete MSFT 5d/10d/20d Discrepancy Trace (`47.89%` / `50.1%` G3 vs `28.40%` / `28.6%` `rolling_mean`)

During the live Render audit (which queries Alpaca IEX split-adjusted bars), `MSFT` 5-day volatility reported **`50.1%` (`gpu_g3`) vs `28.6%` (`rolling_mean`)** (`+21.5 pp`). On our reproducible Yahoo auto-adjusted `MSFT` snapshot at the exact same completed session (`origin_date = 2026-11-20`, `origin_close = $484.52`, `data_fingerprint = 94526b638d5438e595b388023e09ebbec031419aed216a66dfdd38233d56752b`), `MSFT` 5-day volatility reports **`47.89%` (`gpu_g3`) vs `28.40%` (`rolling_mean`)** (`+19.50 pp`) — and rises even further to **`55.64%`** at `H=10` and **`54.15%`** at `H=20`.

### 3.1 Multi-Model Comparison on MSFT (`as_of = 2026-11-20`, `Close = $484.52`)

| Model Identifier | Underlying Formula / Family | `H=5` Cum. Variance | `H=5` Ann. Vol | `H=10` Cum. Variance | `H=10` Ann. Vol | `H=20` Cum. Variance | `H=20` Ann. Vol | `H=5` Risk Level (Ratio vs 60d) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`rolling_mean` (`auto`)** | `H * Var_c2c(60d, ddof=1)` | `0.00159993` | **`28.40%`** | `0.00319985` | **`28.40%`** | `0.00639971` | **`28.40%`** | `Moderate (1.0000x)` |
| **`persistence` (`G0`)** | `H * Var_c2c(20d, ddof=1)` (`panel_rolling_base`) | `0.00286752` | **`38.02%`** | `0.00573504` | **`38.02%`** | `0.01147009` | **`38.02%`** | `Elevated (1.3388x)` |
| **`ewma`** | RiskMetrics `λ = 0.94` (`alpha = 0.06`) | `0.00219860` | **`33.29%`** | `0.00439720` | **`33.29%`** | `0.00879441` | **`33.29%`** | `Elevated (1.1723x)` |
| **`garch_11`** | Causal Gaussian GARCH(1,1) (trailing 252d MLE) | `0.00276618` | **`37.34%`** | `0.00566487` | **`37.78%`** | `0.01174667` | **`38.47%`** | `Elevated (1.3149x)` |
| **`har_rv`** | Recursive `exp(log-HAR)` without Jensen smearing | `0.00045051` | **`15.07%`** | `0.00074495` | **`13.70%`** | `0.00102625` | **`11.37%`** | `Low (0.5306x)` |
| **`gpu_g3`** | `exp(clip(onnx(x) - 0.5 + log(G0_20d), -30, 10))` | `0.00455128` | **`47.89%`** | `0.01228316` | **`55.64%`** | `0.02327088` | **`54.15%`** | `Elevated (1.6866x)` |

### 3.2 Exact Step-by-Step Mathematical Decomposition (`gpu_g3` vs `rolling_mean`)

For horizon $H$, `g3_volatility.g3_cumulative_variance` computes:
$$\widehat{V}_{\text{G3}}(H) = \exp\!\left(\operatorname{clip}\!\left(\text{onnx}(x) - b_0 + \ln B_{20}(H),\,-30.0,\,10.0\right)\right), \qquad \widehat{\sigma}_{\text{G3}}(H) = \sqrt{\frac{252}{H}\widehat{V}_{\text{G3}}(H)}$$
where $b_0 = 0.5$ (`graph_base_score`) and $B_{20}(H) = H \cdot s_{20}^2$ is the **20-day** rolling variance base (`G0`), **not** the 60-day production `rolling_mean` base $B_{60}(H) = H \cdot s_{60}^2$. Consequently, the ratio between G3 and production `rolling_mean` factors exactly into two multiplicative stages:
$$\frac{\widehat{\sigma}_{\text{G3}}(H)}{\widehat{\sigma}_{60}(H)} = \underbrace{\frac{\widehat{\sigma}_{20}(H)}{\widehat{\sigma}_{60}(H)}}_{\text{Stage 1: 20d vs 60d Base Anchor}} \times \underbrace{\exp\!\left(\frac{\delta_H}{2}\right)}_{\text{Stage 2: Learned G3 Tree Multiplier}}, \qquad \delta_H = \text{onnx}_H(x) - 0.5$$

| Decomposition Step | Quantity / Formula | `H=5` Value | `H=10` Value | `H=20` Value |
| :--- | :--- | :---: | :---: | :---: |
| **1. Production 60d Base** | $B_{60}(H) = H \cdot s_{60}^2$ | `0.00159993` | `0.00319985` | `0.00639971` |
| **2. Production 60d Ann. Vol** | $\widehat{\sigma}_{60} = \sqrt{(252/H) B_{60}(H)}$ | **`28.40%`** | **`28.40%`** | **`28.40%`** |
| **3. G3 Base Anchor (`G0` 20d)** | $B_{20}(H) = H \cdot s_{20}^2$ (`panel_rolling_base`) | `0.00286752` | `0.00573504` | `0.01147009` |
| **4. G3 Base Anchor Ann. Vol** | $\widehat{\sigma}_{20} = \sqrt{(252/H) B_{20}(H)}$ | **`38.02%`** | **`38.02%`** | **`38.02%`** |
| **5. Stage 1 Base Shift (`20d / 60d`)** | Variance ratio $B_{20}/B_{60}$ / Vol ratio $\widehat{\sigma}_{20}/\widehat{\sigma}_{60}$ | `1.7923x` / **`1.3388x` (`+9.62 pp`)** | `1.7923x` / **`1.3388x` (`+9.62 pp`)** | `1.7923x` / **`1.3388x` (`+9.62 pp`)** |
| **6. Raw ONNX Margin** | $\text{onnx}_H(x)$ | `+0.961961` | `+1.261634` | `+1.207461` |
| **7. Learned Log-Var Correction** | $\delta_H = \text{onnx}_H(x) - 0.5$ | **`+0.461961`** | **`+0.761634`** | **`+0.707461`** |
| **8. Stage 2 Learned Multiplier** | Variance $\exp(\delta_H)$ / Vol $\exp(\delta_H / 2)$ | `1.5872x` / **`1.2598x` (`+9.88 pp`)** | `2.1418x` / **`1.4635x` (`+17.62 pp`)** | `2.0288x` / **`1.4244x` (`+16.13 pp`)** |
| **9. Complete Log-Variance** | $\ln B_{20}(H) + \delta_H$ (Clip `[-30, 10]` binding?) | `-5.3923` (`False`) | `-4.3995` (`False`) | `-3.7606` (`False`) |
| **10. Final G3 Forecast** | $\widehat{V}_{\text{G3}}(H)$ / $\widehat{\sigma}_{\text{G3}}(H)$ | `0.00455128` / **`47.89%`** | `0.01228316` / **`55.64%`** | `0.02327088` / **`54.15%`** |
| **11. Total Shift (`G3 / 60d`)** | Variance ratio / Vol ratio / Percentage-point gap | `2.8447x` / `1.6866x` / **`+19.50 pp`** | `3.8387x` / `1.9593x` / **`+27.24 pp`** | `3.6362x` / `1.9069x` / **`+25.75 pp`** |
| **12. Log-Vol Attribution Share** | Stage 1 (`20d vs 60d` base) vs Stage 2 (G3 $\delta_H/2$) | **`55.81%` vs `44.19%`** | **`43.38%` vs `56.62%`** | **`45.20%` vs `54.80%`** |
| **13. Accounting Identity Error** | $|(\widehat{\sigma}_{20}/\widehat{\sigma}_{60})\exp(\delta_H/2) - \widehat{\sigma}_{\text{G3}}/\widehat{\sigma}_{60}|$ | `1.11e-15` | `1.11e-15` | `1.55e-15` |

### 3.3 Why $\delta_5$ Swung by `+0.838` Log-Variance Units in Two Sessions (`2026-11-18..2026-11-20`)

| Session Date | MSFT Close | 1d Log Return | 60d `rolling_mean` Ann. Vol | 20d `G0` Base Ann. Vol | G3 Learned $\delta_5$ | G3 Vol Multiplier $\exp(\delta_5/2)$ | Final G3 `H=5` Ann. Vol |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2026-11-16` | `$516.99` | `-1.13%` | `27.51%` | `36.54%` | `-0.1986` | `0.9055x` | `33.08%` |
| `2026-11-17` | `$505.26` | `-2.29%` | `27.90%` | `37.33%` | `-0.0599` | `0.9705x` | `36.22%` |
| `2026-11-18` | `$507.31` | `+0.40%` | `27.86%` | `37.37%` | **`-0.3761`** | **`0.8286x`** | **`30.96%`** |
| `2026-11-19` | `$498.30` | `-1.79%` | `28.05%` | `37.07%` | `+0.0351` | `1.0177x` | `37.73%` |
| `2026-11-20` | `$484.52` | **`-2.80%`** | **`28.40%`** | **`38.02%`** | **`+0.4620`** | **`1.2598x`** | **`47.89%`** |

On `2026-11-20`, MSFT closed down `-2.80%` (6.0th 1-year percentile), pushing `ret_5d` to `-7.62%` (4.0th percentile), `downside_sq` to `7.86e-4` (94.4th percentile), `neg_indicator = 1.0`, `range_park_20` to `3.74e-4` (96.4th percentile), and `yz_chg_5` to `+6.58e-5` (95.6th percentile). Because G3's trees split heavily on single-day downside shock features (`downside_sq`, `ret_1d`, `ret_5d`, `har_log_daily`) *on top of* an already-elevated 20-day base (`38.02%`), G3 compounds the recent sell-off twice: once via the 20d base (`+9.62 pp` over 60d) and a second time via the tree correction (`+9.88 pp` at `H=5`, `+17.62 pp` at `H=10`).

### 3.4 Why `har_rv` (`causal_log_har`) Collapses to `15.07% / 13.70% / 11.37%`

In `backend/services/volatility_snapshot.py:177-199`, `causal_log_har_forecasts` fits ridge-regularized OLS to $\ln(\max(RV_t, 10^{-12}))$ and rolls forward recursively via $\widehat{RV}_{t+k} = \exp(x_{t+k-1}^\top \hat{\beta})$ **without a Jensen's inequality smearing correction** ($\mathbb{E}[\exp(\varepsilon)] > \exp(\mathbb{E}[\varepsilon]) = 1$). Because daily squared returns $RV_t = r_t^2$ have extreme left-tail log-variance dispersion ($\text{Var}(\ln r_t^2) \approx \pi^2/2 \approx 4.93$ for Gaussian returns, where $\exp(\mathbb{E}[\ln r^2]) = \exp(-\gamma - \ln 2)\sigma^2 \approx 0.2807\,\sigma^2$), exponentiating the conditional mean of $\ln RV_t$ estimates the **geometric mean** rather than the **arithmetic mean** of variance. Across all 22,595 OOS origins (Section 4.3), `har_rv`'s mean forecast-to-realized variance ratio is **`0.180` (`H=5`), `0.157` (`H=10`), and `0.128` (`H=20`)** — a structural 5.5x–7.8x downward bias despite having the highest Mincer–Zarnowitz $R^2$ (`0.212 / 0.285 / 0.375`).

---

## 4. Volatility Promotion Gate (`g3_h{5,10,20}.onnx` vs Production Baselines)

### 4.1 Predeclared 5-Clause Promotion Gate (`PROMOTE_G3_TO_AUTO`)

For each horizon $H \in \{5, 10, 20\}$, `PROMOTE_G3_TO_AUTO(H) = PASS` iff all five clauses hold on the purged OOS window (`origin_date >= 2023-11-16`, `target_end_date <= 2026-11-20`):
1. **`qlike_beats_rolling_60d_pooled_holm`**: Pooled mean $\Delta\text{QLIKE}(\text{roll60} - \text{G3}) > 0$, date-aggregated Bartlett HAC (`lag = H-1`) Holm-Bonferroni adjusted $p_{\text{holm}} < 0.05$ across the 6 primary tests, and 95% CIs exclude zero under primary HAC (`H-1`), 2x-lag HAC (`2(H-1)`), and circular date-block bootstrap (`block_length = 2H`, `B = 2,000`).
2. **`qlike_beats_har_rv_pooled_holm`**: Same superiority criteria against production `har_rv` (`causal_log_har`).
3. **`win_rate_ge_70pct_vs_rolling_pooled_us_lse`**: Asset-level win rate (`mean_delta_qlike > 0` among eligible assets with $\ge 250$ common origins) $\ge 70.0\%$ on **Pooled (`>= 21/30`)**, **US (`>= 14/20`)**, and **LSE (`>= 7/10`)** against 60d `rolling_mean`, and $\ge 70.0\%$ on Pooled against `har_rv`.
4. **`mincer_zarnowitz_joint_calibration`**: In annualized variance space $v = (252/H)V$, Driscoll-Kraay date-clustered Bartlett HAC (`lag = H-1`) joint Wald test of $H_0: (\alpha, \beta) = (0, 1)$ either has $p_{\text{wald}} \ge 0.05$ with point estimates inside predeclared equivalence bounds ($|\alpha/\bar{y}| \le 0.15$, $\beta \in [0.80, 1.20]$, $\bar{f}/\bar{y} \in [0.85, 1.15]$), or both 95% HAC CIs lie strictly inside those equivalence bounds.
5. **`subpanel_non_degradation_us_and_lse`**: Both US and LSE sub-panels have positive mean $\Delta\text{QLIKE} > 0$ vs 60d `rolling_mean` with upper 95% HAC CI $> 0$.

### 4.2 Out-of-Sample QLIKE, Bartlett HAC, Block Bootstrap & Win-Rate Table

| Horizon | Panel & Comparator | Common Origins (Dates) | Mean QLIKE Base | Mean QLIKE G3 | Mean `ΔQLIKE` (Rel. Gain) | Bartlett HAC `L = H-1` `t`-stat (`p_raw` / `p_holm`) | HAC `L = H-1` 95% CI | HAC `2L = 2(H-1)` 95% CI | Circular Block Bootstrap (`b = 2H`) 95% CI | Asset Win Rate ($\ge 250$ origins) | Dates Won % |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`H=5`** | **Pooled vs `rolling_mean` (60d)** | `22,595 (770)` | `0.5542` | `0.4950` | **`+0.0592 (+10.7%)`** | `+3.64 (2.76e-04 / 8.27e-04)` | `[+0.0268, +0.0894]` | `[+0.0225, +0.0937]` | `[+0.0261, +0.0997]` | **`24/30 (80.0%)`** | `57.8%` |
| **`H=5`** | **Pooled vs `har_rv`** | `22,595 (770)` | `3.7281` | `0.4950` | **`+3.2331 (+86.7%)`** | `+14.90 (3.49e-50 / 2.10e-49)` | `[+2.7987, +3.6467]` | `[+2.7418, +3.7036]` | `[+2.8025, +3.7774]` | **`30/30 (100.0%)`** | `97.8%` |
| **`H=5`** | *Pooled vs `G0` (20d diagnostic)* | `22,595 (770)` | `0.6344` | `0.4950` | `+0.1394 (+22.0%)` | `+8.63 (6.30e-18 / n/a)` | `[+0.1066, +0.1693]` | `[+0.1031, +0.1729]` | `[+0.1049, +0.1757]` | `30/30 (100.0%)` | `72.5%` |
| **`H=5`** | **US vs `rolling_mean` (60d)** | `15,020 (751)` | `0.5682` | `0.4979` | **`+0.0703 (+12.4%)`** | `+3.63 (2.78e-04 / 8.35e-04)` | `[+0.0324, +0.1082]` | `[+0.0268, +0.1138]` | `[+0.0324, +0.1197]` | **`17/20 (85.0%)`** | `61.4%` |
| **`H=5`** | **LSE vs `rolling_mean` (60d)** | `7,575 (758)` | `0.5264` | `0.4891` | **`+0.0373 (+7.1%)`** | `+2.20 (2.78e-02 / 8.35e-02)` | `[+0.0041, +0.0707]` | `[+0.0003, +0.0745]` | `[+0.0036, +0.0797]` | **`7/10 (70.0%)`** | `47.2%` |
| **`H=10`** | **Pooled vs `rolling_mean` (60d)** | `22,445 (765)` | `0.3910` | `0.3391` | **`+0.0519 (+13.3%)`** | `+2.99 (2.81e-03 / 5.63e-03)` | `[+0.0176, +0.0846]` | `[+0.0163, +0.0858]` | `[+0.0209, +0.0885]` | **`24/30 (80.0%)`** | `57.5%` |
| **`H=10`** | **Pooled vs `har_rv`** | `22,445 (765)` | `4.4124` | `0.3391` | **`+4.0734 (+92.3%)`** | `+13.24 (4.88e-40 / 1.95e-39)` | `[+3.4591, +4.6607]` | `[+3.4239, +4.6959]` | `[+3.5029, +4.7529]` | **`30/30 (100.0%)`** | `99.9%` |
| **`H=10`** | *Pooled vs `G0` (20d diagnostic)* | `22,445 (765)` | `0.4783` | `0.3391` | `+0.1392 (+29.1%)` | `+8.42 (3.62e-17 / n/a)` | `[+0.1057, +0.1697]` | `[+0.1044, +0.1710]` | `[+0.1053, +0.1708]` | `30/30 (100.0%)` | `78.3%` |
| **`H=10`** | **US vs `rolling_mean` (60d)** | `14,920 (746)` | `0.4069` | `0.3452` | **`+0.0618 (+15.2%)`** | `+2.96 (3.12e-03 / 6.23e-03)` | `[+0.0208, +0.1027]` | `[+0.0176, +0.1060]` | `[+0.0238, +0.1102]` | **`17/20 (85.0%)`** | `60.5%` |
| **`H=10`** | **LSE vs `rolling_mean` (60d)** | `7,525 (753)` | `0.3593` | `0.3270` | **`+0.0323 (+9.0%)`** | `+1.81 (6.96e-02 / 1.39e-01)` | **`[-0.0026, +0.0673]`** | **`[-0.0031, +0.0679]`** | **`[-0.0026, +0.0692]`** | **`7/10 (70.0%)`** | `48.9%` |
| **`H=20`** | **Pooled vs `rolling_mean` (60d)** | `22,145 (755)` | `0.2860` | `0.2442` | **`+0.0418 (+14.6%)`** | **`+1.82 (6.86e-02 / 6.86e-02)`** | **`[-0.0032, +0.0872]`** | **`[-0.0090, +0.0930]`** | **`[-0.0016, +0.0994]`** | **`21/30 (70.0%)`** | `53.9%` |
| **`H=20`** | **Pooled vs `har_rv`** | `22,145 (755)` | `5.7907` | `0.2442` | **`+5.5465 (+95.8%)`** | `+13.45 (3.16e-41 / 1.58e-40)` | `[+4.7363, +6.3524]` | `[+4.7017, +6.3870]` | `[+4.8115, +6.4429]` | **`30/30 (100.0%)`** | `100.0%` |
| **`H=20`** | *Pooled vs `G0` (20d diagnostic)* | `22,145 (755)` | `0.3880` | `0.2442` | `+0.1438 (+37.1%)` | `+7.01 (2.40e-12 / n/a)` | `[+0.1038, +0.1845]` | `[+0.1037, +0.1846]` | `[+0.1082, +0.1855]` | `29/30 (96.7%)` | `78.9%` |
| **`H=20`** | **US vs `rolling_mean` (60d)** | `14,720 (736)` | `0.3004` | `0.2503` | **`+0.0502 (+16.7%)`** | `+1.80 (7.23e-02 / 7.23e-02)` | **`[-0.0045, +0.1049]`** | **`[-0.0128, +0.1131]`** | **`[-0.0031, +0.1207]`** | **`15/20 (75.0%)`** | `53.8%` |
| **`H=20`** | **LSE vs `rolling_mean` (60d)** | `7,425 (743)` | `0.2574` | `0.2321` | **`+0.0252 (+9.8%)`** | `+1.09 (2.75e-01 / 2.75e-01)` | **`[-0.0201, +0.0706]`** | **`[-0.0240, +0.0745]`** | **`[-0.0209, +0.0799]`** | **`6/10 (60.0%)`** | `52.0%` |

### 4.3 Joint Mincer–Zarnowitz Calibration Table (Driscoll-Kraay Date-Clustered Bartlett HAC)

Equivalence bounds: $|\alpha_{\text{rel}}| = |\alpha / \bar{y}| \le 0.15$, $\beta \in [0.80, 1.20]$, mean forecast/realized ratio $\bar{f}/\bar{y} \in [0.85, 1.15]$.

| Horizon | Model & Panel | Intercept $\alpha$ | Rel. Intercept $\alpha / \bar{y}$ (`95% HAC CI`) | Slope $\beta$ (`95% HAC CI`) | Mean Ratio $\bar{f} / \bar{y}$ | Joint Wald $\chi^2(2)$ (`p`-value, `L = H-1`) | 2x-Lag Wald $\chi^2(2)$ (`p`-value, `2L`) | $R^2$ | MZ Gate Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`H=5`** | **`gpu_g3` (Pooled)** | `+0.01422` | **`+0.147`** (`[+0.007, +0.287]`) | **`0.853`** (`[0.655, 1.050]`) | **`1.000`** | **`4.73 (9.41e-02)`** | `5.44 (6.60e-02)` | `0.212` | **`PASS`** |
| **`H=5`** | `rolling_mean` 60d (Pooled) | `+0.02442` | `+0.252` (`[+0.178, +0.327]`) | `0.766` (`[0.633, 0.899]`) | `0.976` | `44.23 (2.48e-10)` | `34.72 (2.90e-08)` | `0.168` | `FAIL` |
| **`H=5`** | `har_rv` (Pooled) | `+0.01062` | `+0.110` (`[+0.016, +0.204]`) | **`4.941`** (`[4.031, 5.850]`) | **`0.180`** | `290.74 (7.36e-64)` | `226.64 (6.11e-50)` | `0.212` | `FAIL` |
| **`H=5`** | `G0` 20d (Pooled) | `+0.03979` | `+0.411` (`[+0.322, +0.500]`) | `0.598` (`[0.454, 0.742]`) | `0.984` | `81.86 (1.67e-18)` | `61.39 (4.67e-14)` | `0.160` | `FAIL` |
| **`H=5`** | *`gpu_g3` (US Sub-Panel)* | `+0.01937` | `+0.168` (`[+0.015, +0.322]`) | `0.880` (`[0.669, 1.091]`) | `0.945` | `9.17 (1.02e-02)` | `10.15 (6.25e-03)` | `0.207` | `FAIL` |
| **`H=5`** | *`gpu_g3` (LSE Sub-Panel)* | `+0.03546` | **`+0.586`** (`[+0.354, +0.819]`) | **`0.342`** (`[0.095, 0.590]`) | **`1.210`** | **`84.54 (4.39e-19)`** | `75.91 (3.28e-17)` | `0.072` | **`FAIL`** |
| **`H=10`** | **`gpu_g3` (Pooled)** | `+0.01820` | **`+0.188`** (`[+0.065, +0.311]`) | **`0.796`** (`[0.629, 0.963]`) | **`1.020`** | **`8.95 (1.14e-02)`** | `7.90 (1.93e-02)` | `0.254` | **`FAIL`** |
| **`H=10`** | `rolling_mean` 60d (Pooled) | `+0.02619` | `+0.270` (`[+0.188, +0.352]`) | `0.750` (`[0.601, 0.899]`) | `0.973` | `42.67 (5.43e-10)` | `30.28 (2.66e-07)` | `0.239` | `FAIL` |
| **`H=10`** | `har_rv` (Pooled) | `+0.01671` | `+0.172` (`[+0.085, +0.259]`) | **`5.272`** (`[4.275, 6.268]`) | **`0.157`** | `217.92 (4.77e-48)` | `184.42 (9.00e-41)` | `0.285` | `FAIL` |
| **`H=10`** | `G0` 20d (Pooled) | `+0.04257` | `+0.439` (`[+0.334, +0.544]`) | `0.574` (`[0.412, 0.736]`) | `0.977` | `67.36 (2.36e-15)` | `48.77 (2.57e-11)` | `0.214` | `FAIL` |
| **`H=10`** | *`gpu_g3` (US Sub-Panel)* | `+0.02182` | `+0.189` (`[+0.056, +0.323]`) | `0.848` (`[0.669, 1.026]`) | `0.956` | `12.45 (1.98e-03)` | `10.97 (4.15e-03)` | `0.254` | `FAIL` |
| **`H=10`** | *`gpu_g3` (LSE Sub-Panel)* | `+0.04225` | **`+0.697`** (`[+0.556, +0.838]`) | **`0.240`** (`[0.104, 0.377]`) | **`1.259`** | **`203.71 (5.83e-45)`** | `164.08 (2.35e-36)` | `0.062` | **`FAIL`** |
| **`H=20`** | **`gpu_g3` (Pooled)** | `+0.00697` | **`+0.072`** (`[-0.061, +0.205]`) | **`0.875`** (`[0.696, 1.055]`) | **`1.060`** | **`1.85 (3.96e-01)`** | `2.28 (3.20e-01)` | `0.325` | **`PASS`** |
| **`H=20`** | `rolling_mean` 60d (Pooled) | `+0.02866` | `+0.297` (`[+0.203, +0.391]`) | `0.718` (`[0.548, 0.889]`) | `0.979` | `38.86 (3.64e-09)` | `26.11 (2.14e-06)` | `0.320` | `FAIL` |
| **`H=20`** | `har_rv` (Pooled) | `+0.02144` | `+0.222` (`[+0.133, +0.312]`) | **`6.054`** (`[4.886, 7.223]`) | **`0.128`** | `205.56 (2.31e-45)` | `176.03 (5.97e-39)` | `0.375` | `FAIL` |
| **`H=20`** | `G0` 20d (Pooled) | `+0.04667` | `+0.484` (`[+0.360, +0.607]`) | `0.527` (`[0.351, 0.703]`) | `0.979` | `59.32 (1.31e-13)` | `51.18 (7.70e-12)` | `0.263` | `FAIL` |
| **`H=20`** | *`gpu_g3` (US Sub-Panel)* | `+0.00984` | `+0.086` (`[-0.061, +0.233]`) | `0.930` (`[0.735, 1.125]`) | `0.983` | `2.51 (2.85e-01)` | `2.97 (2.27e-01)` | `0.327` | `PASS` |
| **`H=20`** | *`gpu_g3` (LSE Sub-Panel)* | `+0.03854` | **`+0.635`** (`[+0.500, +0.770]`) | **`0.271`** (`[0.146, 0.396]`) | **`1.345`** | **`139.31 (5.60e-31)`** | `116.89 (4.15e-26)` | `0.072` | **`FAIL`** |

### 4.4 Structural Diagnosis of LSE Overprediction (`meanRatio = 1.210..1.345`, `slope = 0.240..0.342`)

Why does `gpu_g3` systematically overpredict LSE realized variance across **10/10 LSE tickers** (`g3 / y` ranging from `1.126x` on `BARC.L` to `1.363x` on `SHEL.L` at `H=5`, whereas 60d `rolling_mean` has `r60 / y = 0.997x` on LSE)?
1. **Unconditional Cross-Market Level Pull:** Over `2023-11-16..2026-11-20`, LSE mega-caps had nearly **half** the realized annualized variance of US equities (`0.0605` LSE vs `0.1151` US). Because `G3` dropped `market_US`/`market_UK` and splits directly on unnormalized variance levels (`rv_5..60`, `har_log_monthly`, `yz_now`), its global trees interpret low-variance LSE sessions as temporarily depressed US stocks and apply an average positive log-variance shift of $\bar{\delta} = +0.2982$ (`H=5`), $+0.3420$ (`H=10`), and $+0.4390$ (`H=20`) — compared to $+0.1610 / +0.2015 / +0.2838$ on US equities.
2. **Shock Overshoot During Market Stress:** During the April 2025 tariff sell-off (`2025-04-07..2025-04-11`), `BP.L` received $\delta_5 = +1.2609$ (`3.53x` variance boost on top of its 20d base), producing an annualized variance forecast of `0.9255` (`96.2%` annualized vol) when subsequent 5-day realized annualized variance was `0.0870` (`29.5%` vol).
3. **Asymmetry of QLIKE vs Mincer–Zarnowitz:** Because $\text{QLIKE}(y, f) = y/f - \ln(y/f) - 1$ penalizes overprediction ($f > y$) much more gently than underprediction ($f < y$), a $+21\%$ to $+35\%$ upward bias still achieves a tiny positive $\Delta\text{QLIKE}$ (`+0.025..+0.037`) while failing Mincer–Zarnowitz calibration catastrophically (`W = 84.5..203.7, p <= 4.39e-19`).

### 4.5 Strictly Post-Freeze Sub-Window Diagnostic (`2026-09-08..2026-11-13`)

| Horizon | Post-Freeze Origin Span | Common Origins (Dates) | Mean `ΔQLIKE` vs 60d `rolling_mean` | Bartlett HAC `L = H-1` `p_two_sided` | HAC `L = H-1` 95% CI | Asset Win Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`H=5`** | `2026-09-08..2026-11-13` | `1,470 (49)` | `+0.1069` | `2.05e-02` | `[+0.0165, +0.1974]` | `20/30 (66.7%)` |
| **`H=10`** | `2026-09-08..2026-11-06` | `1,320 (44)` | `+0.0901` | `6.98e-03` | `[+0.0246, +0.1556]` | `22/30 (73.3%)` |
| **`H=20`** | `2026-09-08..2026-10-23` | `1,020 (34)` | `+0.0436` | `8.76e-06` | `[+0.0244, +0.0628]` | `24/30 (80.0%)` |

### 4.6 Volatility Promotion Gate Matrix (`PROMOTE_G3_TO_AUTO`)

| Predeclared Gate Clause | `H=5` | `H=10` | `H=20` |
| :--- | :---: | :---: | :---: |
| 1. `qlike_beats_rolling_60d_pooled_holm` ($p_{\text{holm}} < 0.05$, HAC `L`/`2L` & Boot CIs $> 0$) | `PASS` (`p_holm = 8.27e-04`) | `PASS` (`p_holm = 5.63e-03`) | **`FAIL` (`p_holm = 0.0686`, CIs `< 0`)** |
| 2. `qlike_beats_har_rv_pooled_holm` ($p_{\text{holm}} < 0.05$, HAC `L`/`2L` & Boot CIs $> 0$) | `PASS` (`p_holm = 2.10e-49`) | `PASS` (`p_holm = 1.95e-39`) | `PASS` (`p_holm = 1.58e-40`) |
| 3. `win_rate_ge_70pct_vs_rolling_pooled_us_lse` (Pooled, US, LSE $\ge 70\%$) | `PASS` (`80% / 85% / 70%`) | `PASS` (`80% / 85% / 70%`) | **`FAIL` (`70% / 75% / 60% LSE`)** |
| 4. `mincer_zarnowitz_joint_calibration` (Pooled Driscoll-Kraay HAC Wald & bounds) | `PASS`* (`W = 4.73, p = 0.0941`) | **`FAIL` (`a_rel = +0.188, b = 0.796, p = 0.0114`)** | `PASS`* (`W = 1.85, p = 0.3958`) |
| 5. `subpanel_non_degradation_us_and_lse` (US & LSE $\Delta\text{QLIKE} > 0$) | `PASS` (`+0.0703 / +0.0373`) | `PASS` (`+0.0618 / +0.0323`) | `PASS` (`+0.0502 / +0.0252`) |
| **Per-Horizon Binary Verdict (`PROMOTE_G3_TO_AUTO`)** | **`PASS`*** *(LSE MZ fails: `W=84.5`)* | **`FAIL`** | **`FAIL`** |

---

## 5. Price Forecast Conformal Calibration & Persistence MAE Gate (`simple_forecast.py`)

### 5.1 Walk-Forward Protocol & Zero-Leakage Verification

At every out-of-sample origin $t \in [\text{2023-11-16}, \text{2026-11-11}]$ (purged at the right edge so all 7 future sessions $t+1\dots t+7$ mature on or before `2026-11-20`), `exact_served_ridge_at_origin` executes the exact production `simple_forecast.py` Ridge pipeline (`make_pipeline(RobustScaler(), Ridge(alpha=8.0))`) using **only** completed sessions $\le t$:
- **Target Maturity Discipline:** A historical origin $s$ has its 7-day target vector matured at origin $t$ if and only if $s + 7 \le t$. All origins $s > t - 7$ are excluded from both the 70/15/15 split and the production refit.
- **Verified Zero Leakage & Serving Parity:** Mutating future bars $s > t$ produces `0.0` change in predictions or bands (`test_exact_served_ridge_zero_future_leakage_and_serving_parity`), feature causality diff is `0.0`, and closed-form Ridge matches `simple_forecast.train_and_forecast(..., model_name="ridge")` within `1e-12`.
- **Causal Volatility Regimes:** Each origin $(i, t)$ is classified into **Elevated** ($\text{vol}_{20}(t) / \text{vol}_{60}(t) > 1.15$, matching `live_volatility.py`'s frozen threshold) or **Calm** ($\le 1.15$) using only trailing returns through $t$.

### 5.2 Empirical Coverage of Served Ridge Uncertainty Band vs Online Conformal Diagnostics

Nominal target is **`80.0%`** (`[q10, q90]`); predeclared acceptable calibration band is **`[75.0%, 85.0%]`** with tail miss bounds **`[5.0%, 15.0%]`**.

| Evaluation Slice | Origins (Dates) | Served Band Pointwise Marginal Coverage | Served Band Simultaneous 7d Path Coverage | Served Lower Tail Miss (`< q10`) | Served Upper Tail Miss (`> q90`) | Served Mean Width (% of Origin) | Diagnostic: Matured Online Pointwise Conformal (Pw / Sim7d) | Diagnostic: Matured Online Simultaneous 7d Conformal (Pw / Sim7d / Width) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled (30 Tickers)** | `22,535 (768)` | **`75.87%`** (`PASS`) | **`45.65%`** | `11.71%` | `12.42%` | `8.01%` | **`77.95%`** / `47.34%` | `92.23%` / **`77.32%`** / `12.66%` |
| **US Panel (20 Tickers)** | `14,980 (749)` | **`76.86%`** (`PASS`) | **`47.70%`** | `11.58%` | `11.57%` | `8.81%` | **`78.08%`** / `47.84%` | `92.31%` / **`77.38%`** / `13.67%` |
| **LSE Panel (10 Tickers)** | `7,555 (756)` | **`73.90%` (`FAIL < 75%`)** | **`41.59%`** | `11.98%` | `14.12%` | `6.42%` | **`77.68%`** / `46.33%` | `92.07%` / **`77.18%`** / `10.67%` |
| **Calm Regime ($\le 1.15$)** | `17,088 (768)` | **`76.87%`** (`PASS`) | **`47.14%`** | `11.36%` | `11.77%` | `8.05%` | **`78.75%`** / `48.74%` | `92.79%` / **`78.63%`** / `12.73%` |
| **Elevated Regime ($> 1.15$)** | `5,447 (729)` | **`72.71%` (`FAIL < 75%`)** | **`40.98%`** | `12.82%` | `14.47%` | `7.87%` | **`75.42%`** / `42.92%` | `90.47%` / **`73.20%`** / `12.45%` |
| *US × Calm* | `11,284 (747)` | `77.87%` | `49.02%` | `11.40%` | `10.73%` | `8.86%` | — | — |
| *US × Elevated* | `3,696 (672)` | **`73.77%` (`< 75%`)** | `43.70%` | `12.10%` | `14.12%` | `8.65%` | — | — |
| *LSE × Calm* | `5,804 (756)` | **`74.94%` (`< 75%`)** | `43.50%` | `11.26%` | `13.80%` | `6.47%` | — | — |
| *LSE × Elevated* | `1,751 (589)` | **`70.45%` (`< 75%`)** | **`35.24%`** | `14.35%` | **`15.20%`** | `6.24%` | — | — |

### 5.3 Per-Horizon Pointwise Coverage & Walk-Forward Persistence MAE Ratio (`h = 1..7`)

| Horizon Day | Served Band Pointwise Coverage | Lower Tail Miss | Upper Tail Miss | Mean Band Width (% of Origin) | Served Ridge MAE (%) | Persistence MAE (%) | Scale-Independent MAE Ratio (`Model / Pers`) | Date-Aggregated Bartlett HAC (`L = h-1`) `t`-stat (`Pers - Model`) | HAC `p_two_sided` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`h = 1`** | `76.96%` (`PASS`) | `11.30%` | `11.73%` | `3.93%` | `1.3303%` | `1.3107%` | **`1.0150`** (`> 1.0`) | `-6.88` | `5.83e-12` |
| **`h = 2`** | `76.62%` (`PASS`) | `11.10%` | `12.28%` | `5.75%` | `1.9336%` | `1.9011%` | **`1.0171`** (`> 1.0`) | `-7.41` | `1.28e-13` |
| **`h = 3`** | `76.34%` (`PASS`) | `11.47%` | `12.19%` | `7.16%` | `2.3977%` | `2.3566%` | **`1.0175`** (`> 1.0`) | `-6.82` | `9.12e-12` |
| **`h = 4`** | `75.58%` (`PASS`) | `11.68%` | `12.74%` | `8.32%` | `2.7974%` | `2.7515%` | **`1.0167`** (`> 1.0`) | `-5.44` | `5.22e-08` |
| **`h = 5`** | `75.31%` (`PASS`) | `11.95%` | `12.74%` | `9.35%` | `3.1418%` | `3.0937%` | **`1.0155`** (`> 1.0`) | `-4.77` | `1.84e-06` |
| **`h = 6`** | `75.44%` (`PASS`) | `12.08%` | `12.48%` | `10.34%` | `3.4623%` | `3.4011%` | **`1.0180`** (`> 1.0`) | `-5.09` | `3.61e-07` |
| **`h = 7`** | **`74.80%` (`FAIL < 75%`)** | `12.39%` | `12.80%` | `11.22%` | `3.7606%` | `3.6931%` | **`1.0183`** (`> 1.0`) | `-4.90` | `9.63e-07` |
| **All `1..7` Pooled** | **`75.87%`** | `11.71%` | `12.42%` | `8.01%` | **`2.6891%`** | **`2.6440%`** | **`1.0171`** (`Boot CI [1.0120, 1.0222]`) | **`HAC(6) t = -6.24` (`HAC(12) t = -6.38`)** | **`4.36e-10`** |
| **US `1..7` Pooled** | `76.86%` | `11.58%` | `11.57%` | `8.81%` | `2.8843%` | `2.8461%` | **`1.0134`** (`Boot CI [1.0063, 1.0201]`) | `HAC(6) t = -3.70` (`HAC(12) t = -3.67`) | `2.13e-04` |
| **LSE `1..7` Pooled** | **`73.90%`** | `11.98%` | `14.12%` | `6.42%` | `2.3021%` | `2.2432%` | **`1.0263`** (`Boot CI [1.0184, 1.0348]`) | `HAC(6) t = -6.65` (`HAC(12) t = -6.53`) | `2.93e-11` |

### 5.4 Price Calibration Gate Matrix (`UPGRADE_PRICE_BAND_LABEL_TO_CALIBRATED`)

| Predeclared Gate Clause | Criterion | Observed Value | Clause Verdict |
| :--- | :--- | :---: | :---: |
| 1. `pooled_pointwise_in_75_85` | Pooled pointwise marginal coverage $\in [75.0\%, 85.0\%]$ | `75.87%` | `PASS` |
| 2. `all_horizons_1_to_7_in_75_85` | Every horizon $h \in \{1\dots 7\}$ coverage $\in [75.0\%, 85.0\%]$ | `h=7 is 74.80%` | **`FAIL`** |
| 3. `us_and_lse_pointwise_in_75_85` | Both US and LSE pointwise coverage $\in [75.0\%, 85.0\%]$ | `US = 76.86%, LSE = 73.90%` | **`FAIL`** |
| 4. `calm_and_elevated_regimes_in_75_85` | Both Calm and Elevated regime coverage $\in [75.0\%, 85.0\%]$ | `Calm = 76.87%, Elevated = 72.71%` | **`FAIL`** |
| 5. `tail_symmetry_5_to_15_each_tail` | Pooled lower and upper tail miss rates $\in [5.0\%, 15.0\%]$ | `Lower = 11.71%, Upper = 12.42%` | `PASS` |
| **Overall Binary Verdict** | **`UPGRADE_PRICE_BAND_LABEL_TO_CALIBRATED`** | **3 Clauses Failed (plus `45.65%` simultaneous 7d coverage)** | **`FAIL`** |

---

## 6. Binding Architectural Recommendations for Milestone 05 Part 2

1. **Keep `rolling_mean` (60d) as the `auto` Volatility Default (`PROMOTE_G3_TO_AUTO = FAIL`):**
   - Retain `AUTO_MODEL_POLICY = {1: "garch_11", 5: "rolling_mean", 10: "rolling_mean", 20: "rolling_mean"}` (`VOLATILITY_DEFAULT_POLICY=v3_baseline`).
   - In Milestone 05 Part 2 (`backend/services/live_volatility.py`, `backend/services/volatility_contract.py`, `scripts/collect_live_forecasts.py`), fix the `live_volatility.py:296` dispatch ordering bug (resolve `auto` to its effective model *before* branching on `GPU_G3_MODEL`) and decouple the V1 collector policy from `AUTO_MODEL_POLICY`.
   - Before any future G3 candidate is re-submitted to this promotion gate, retrain G3 using scale-free / market-normalized features (so low-variance UK LSE equities are not pulled upward by `+21%..+35%` toward US variance levels) and re-anchor G3's base margin on 60-day `rolling_mean` rather than 20-day `G0`.
2. **Retain M4's Honest Price Range Label (`UPGRADE_PRICE_BAND_LABEL_TO_CALIBRATED = FAIL`):**
   - Keep the chart uncertainty band labeled **`Historical error range (experimental)`**. Do **not** upgrade the label to `"Calibrated 80% range"`.
   - When conformal recalibration is revisited in a future milestone, replace `simple_forecast.py`'s static 70/15 validation residual split with strict target-matured trailing online split conformal (which our diagnostic proves restores pointwise coverage to `77.95%` pooled, `77.68%` LSE, and `75.42%` elevated) or volatility-scaled simultaneous 7-day conformal (`77.32%` simultaneous path coverage).
3. **Do Not Promote Learned Price Models Over Persistence (`PRICE_RIDGE_BEATS_PERSISTENCE_MAE = False`):**
   - Across `22,535` walk-forward origins, Ridge has a scale-independent MAE ratio of `1.0171` vs persistence (`p = 4.36e-10`). Continue framing the 7-session price trajectory as an experimental benchmark rather than a persistence-beating alpha signal.
