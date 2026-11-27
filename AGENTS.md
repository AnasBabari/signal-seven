# AGENTS.md — Signal Seven Contributor & Agent Guide

This repository contains **Signal Seven**, a full-stack equity forecasting and volatility-analytics web application covering a **286-ticker US (NASDAQ/NYSE) and UK (LSE) universe**.

---

## Repository Layout

| Path | Responsibility | Runtime / Stack |
| :--- | :--- | :--- |
| `backend/` | FastAPI API, market-data adapters, Ridge + conformal price forecasts, causal Duan-smeared HAR-RV / GARCH / G3 ONNX volatility forecasting, news & SEC EDGAR aggregation, PostgreSQL/SQLite forecast ledger. | Python 3.11, `uv` (`backend/uv.lock`), Render Free (`0.1 vCPU / 512 MB`) |
| `frontend/` | React + Vite + Chart.js web app (Trading 212-inspired dark UI, lazy-loaded chart engine, local 286-ticker search, Overview/News/Performance tabs). | Node 22.x (`>=22.12`), npm (`frontend/package-lock.json`), Vercel |
| `research/` | Offline ML training, GPU pipelines, M5 statistical validation gates (HAC, Mincer–Zarnowitz, conformal calibration), and research tests. | Python 3.11 + CPU/CUDA PyTorch (`torch==2.11.0`) |
| `scripts/` | Deterministic catalog generation, offline P/E batch refresh, live forecast collection/settlement/export CLI. | Python 3.11 |
| `.github/workflows/` | CI (`ci.yml`), live ledger collection/settlement/export (`live-forecast-operations.yml`), weekly offline P/E refresh (`refresh-fundamentals.yml`). | GitHub Actions (`ubuntu-24.04`) |

---

## Local Setup & Verification Commands

Match `.github/workflows/ci.yml` before opening or updating any PR.

### Backend & Research
```bash
# 1. Verify lockfile & sync locked environment
uv lock --project backend --check
uv sync --project backend --locked --no-install-project

# 2. Lint, format, security, and core typecheck
backend/.venv/bin/ruff check .
backend/.venv/bin/ruff format --check .
backend/.venv/bin/bandit -r backend -x backend/tests,backend/venv,backend/.venv -ll
backend/.venv/bin/mypy --config-file backend/pyproject.toml \
  backend/api.py backend/config.py backend/calendars.py \
  backend/routes/health.py backend/routes/common.py backend/routes/simple_forecast.py

# 3. Tests (CI installs CPU torch for research/tests/test_price_gpu_pipeline.py)
backend/.venv/bin/pytest -c backend/pyproject.toml backend/tests research/tests -q -ra
```
*(On Windows PowerShell, use `.\backend\.venv\Scripts\<tool>.exe` and `;` instead of `&&`. If `torch` is not installed in `backend/.venv`, add `--ignore=research/tests/test_price_gpu_pipeline.py` locally and rely on CI for the PyTorch test.)*

### Frontend
```bash
cd frontend
npm ci
npm test -- --run
npm run build
npm run check:production-bundle
npm run e2e -- e2e/server-contract.spec.js e2e/audit-contract.spec.js
```

---

## Architecture Invariants & Guardrails

### 1. 286-Ticker Catalog Determinism & Offline Fundamentals
- `data/tri_exchange/manifest_broad_300.json`, `backend/data/tri_exchange/manifest_broad_300.json`, `backend/data/universe_metadata.json`, and `frontend/src/universe.js` **must remain in exact parity**.
- `backend/tests/test_universe_catalog.py` asserts byte-for-byte equality between `universe_metadata.json` / `universe.js` and `scripts/generate_universe_metadata.py`.
- **Never** write dynamic/time-varying fundamentals (such as P/E ratios) into `universe_metadata.json`; store offline-refreshed fundamentals in `backend/data/universe_fundamentals.json` via `scripts/refresh_universe_pe.py` (which enforces `1.0 <= pe_ttm < 1000.0`).
- **Never** call slow `yfinance.Ticker.info` on live FastAPI request paths (`/api/v1/info` uses `fast_info` + mtime-cached disk metadata/fundamentals + 60s negative caching).

### 2. London (LSE) Quote Units & Currency Overrides
- Most `.L` stocks quote share prices in **GBp (pence)** while market cap is normalized to **GBP (`£`)** (guarding against Yahoo `fast_info` returning `shares * pence` 100× inflated).
- **Exceptions:** `CPG.L` and `IHG.L` are USD-quoted on LSE (`backend/market_data/quote_units.py`); do not apply pence scaling to them.

### 3. ML Causality, Serving Parity & Statistical Honesty
- **Volatility forecasting:** Default policy is `v5_duan_har` (`1d: garch_11`, `5d/10d/20d: har_rv` with strictly causal in-sample Duan smearing in `backend/services/volatility_snapshot.py`). Preserve instant rollback via `VOLATILITY_DEFAULT_POLICY=v3_baseline`, and keep `LIVE_MODEL_POLICY_V1` pinned to `rolling_mean` for live ledger comparability.
- **Price uncertainty calibration:** Served Ridge intervals in `backend/services/simple_forecast.py` use strictly target-matured (`origin_pos + h <= latest_pos`) volatility-scaled online split conformal (`blend_ewma94_v60`).
  - Keep `research/validation/validate_m5_models.py::exact_served_ridge_at_origin` at `1e-12` parity with `train_and_forecast(..., model_name="ridge")` (`research/tests/test_m5_validation_gates.py`).
  - Never label an interval as `"calibrated"` or `"80% pointwise conformal range"` in the UI unless `interval_calibration.m5_gate_verdict === "PASS"`; fall back to `"Historical error range (experimental)"` otherwise.
- **News context:** Financial news and SEC EDGAR filings are aggregated for market context and sentiment display without mutating M5-validated point/conformal model predictions. Keep SEC EDGAR requests strictly under 10 req/s with a compliant `User-Agent`.

### 4. Production Constraints (Render Free & Vercel)
- **Render Free (`0.1 vCPU / 512 MB`):** Keep `FORECAST_AUTO_CANDIDATES=ridge` for deterministic, fast cold fits. Do not re-add unused heavy runtime dependencies (`boto3`, `cryptography`, `filelock`, `pyarrow`) to `[project].dependencies` in `backend/pyproject.toml` (`backend/tests/test_toolchain_parity.py` enforces this, keeping `pyarrow` in `[dependency-groups].dev`).
- Always use `uv sync --locked` (never `--frozen`) in Dockerfiles, `render.yaml`, and workflows.
- **Never** expose API keys or collector tokens in `VITE_*` variables or commit `.env` files.

### 5. Frontend Bundle Ceilings & E2E Contracts
- `frontend/scripts/check-production-bundle.mjs` fails CI if:
  - Lazy `chart-engine*.js` exceeds **`175,000 B`**
  - Total CSS exceeds **`36,000 B`**
  - Dead `[data-theme="light"]`, TFJS, or `jszip` code re-enters the bundle
- Preserve all 16 Playwright contracts in `frontend/e2e/server-contract.spec.js` and `frontend/e2e/audit-contract.spec.js`, including:
  - Sticky `.topbar` with computed background alpha `>= 0.85`
  - Mobile `390×844` autocomplete anchored with `0px` overlap below `View outlook` and `0px` horizontal overflow
  - Accessible keyboard-only range switching, modal Escape/focus trapping, and opt-in `Table` fallback on chart chunk failure
- **Cross-platform `package-lock.json`:** When updating npm dependencies on Windows, verify Linux optional dependencies (e.g., `@emnapi/runtime`) remain intact so `npm ci` passes on `ubuntu-24.04`.

---

## Git & Pull Request Workflow

- `main` is protected (`backend-and-research` and `frontend` checks required; no direct pushes or force pushes). Always create a feature branch and open a PR.
- Stage files explicitly (`git add <paths>`) — never use `git add -A` so local scratch files or untracked artifacts are not accidentally committed.
- On Windows PowerShell, pass multiline PR descriptions via `gh pr create --body-file <file>` / `gh pr edit --body-file <file>` to avoid backtick escape corruption.
