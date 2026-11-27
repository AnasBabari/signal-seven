# AGENTS.md — Signal Seven Contributor & Agent Guide

This repository contains **Signal Seven**, a full-stack equity forecasting and volatility-analytics web application covering a **286-ticker US (NASDAQ/NYSE) and UK (LSE) universe**.

---

## Repository Layout

| Path | Responsibility | Runtime / Stack |
| :--- | :--- | :--- |
| `backend/` | FastAPI API, market-data adapters, Ridge + conformal price forecasts, causal Duan-smeared HAR-RV / GARCH / G3 ONNX volatility forecasting, news & SEC EDGAR aggregation, PostgreSQL/SQLite forecast ledger. | Python 3.11, `uv` (`backend/uv.lock`), Render Free (`0.1 vCPU / 512 MB`) |
| `frontend/` | React + Vite + Chart.js web app (Trading 212-inspired dark UI, lazy-loaded chart engine, local 286-ticker search, Overview/News/Performance tabs). | Node 22.22.2 or later in the 22.x series, npm (`frontend/package-lock.json`), Vercel |
| `research/` | Offline ML training, GPU pipelines, M5 statistical validation gates (HAC, Mincer–Zarnowitz, conformal calibration), and research tests. | Python 3.11 + CPU/CUDA PyTorch (`torch==2.11.0`) |
| `scripts/` | Deterministic catalog generation, offline P/E batch refresh, live forecast collection/settlement/export CLI. | Python 3.11 |
| `.github/workflows/` | CI (`ci.yml`), live ledger collection/settlement/export (`live-forecast-operations.yml`), weekly offline P/E refresh (`refresh-fundamentals.yml`). | GitHub Actions (`ubuntu-24.04`) |

---

## Local Setup & Verification Commands

Match `.github/workflows/ci.yml` before opening or updating any PR.

### Backend & Research

Run these commands from the repository root in Bash or PowerShell with Python 3.11 available. Check each command succeeds before continuing.

```text
python -m pip install uv==0.11.16
uv lock --project backend --check
uv sync --project backend --python 3.11 --locked --no-install-project
uv pip install --python backend/.venv "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu
uv run --project backend --no-sync ruff check .
uv run --project backend --no-sync ruff format --check .
uv run --project backend --no-sync bandit -r backend -x backend/tests,backend/venv,backend/.venv -ll
uv run --project backend --no-sync mypy --config-file backend/pyproject.toml backend/api.py backend/config.py backend/calendars.py backend/routes/health.py backend/routes/common.py backend/routes/simple_forecast.py
uv run --project backend --no-sync pytest -c backend/pyproject.toml backend/tests research/tests -q -ra
```

### Frontend
```bash
cd frontend
npm ci
npm test -- --run
npm run build
npm run check:production-bundle
npm run e2e:install
npm run e2e -- e2e/server-contract.spec.js e2e/audit-contract.spec.js
```

---

## Architecture Invariants & Guardrails

### 1. 286-Ticker Catalog Determinism & Offline Fundamentals
- `data/tri_exchange/manifest_broad_300.json`, `backend/data/tri_exchange/manifest_broad_300.json`, `backend/data/universe_metadata.json`, and `frontend/src/universe.js` **must remain in exact parity**.
- `backend/tests/test_universe_catalog.py` compares the ticker lists across both manifests and checks each generated file against the output of `scripts/generate_universe_metadata.py`.
- **Never** write dynamic/time-varying fundamentals (such as P/E ratios) into `universe_metadata.json`; store offline-refreshed fundamentals in `backend/data/universe_fundamentals.json` via `scripts/refresh_universe_pe.py` (which enforces `1.0 <= pe_ttm < 1000.0`).
- `/api/v1/info` reads `fast_info` first and falls back to `Ticker.info` when the core fast metrics are all missing. Use the existing disk fundamentals fallback for missing P/E values.

### 2. London (LSE) Quote Units & Currency Overrides
- Most `.L` stocks quote share prices in **GBp (pence)** while market cap is normalized to **GBP (`£`)** (guarding against Yahoo `fast_info` returning `shares * pence` 100× inflated).
- **Exceptions:** `CPG.L` and `IHG.L` are USD-quoted on LSE (`backend/market_data/quote_units.py`); do not apply pence scaling to them.

### 3. ML Causality, Serving Parity & Statistical Honesty
- The default volatility policy `v5_duan_har` uses `garch_11` at one session and `har_rv` at 5, 10 and 20 sessions with causal Duan smearing. Keep `VOLATILITY_DEFAULT_POLICY=v3_baseline` available for rollback. The frozen `LIVE_MODEL_POLICY_V1` ledger uses `garch_11` at one session and `rolling_mean` at 5, 10 and 20.
- **Price uncertainty calibration:** Served Ridge intervals in `backend/services/simple_forecast.py` use strictly target-matured (`origin_pos + h <= latest_pos`) volatility-scaled online split conformal (`blend_ewma94_v60`).
  - Keep `research/validation/validate_m5_models.py::exact_served_ridge_at_origin` at `1e-12` parity with `train_and_forecast(..., model_name="ridge")` (`research/tests/test_m5_validation_gates.py`).
  - Display `"80% pointwise conformal range"` only when `interval_calibration.m5_gate_verdict === "PASS"` and `interval_calibration.method === "volatility_scaled_online_split_conformal"`. Otherwise use `"Historical error range (experimental)"`.
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
- The Playwright contracts in `frontend/e2e/server-contract.spec.js` and `frontend/e2e/audit-contract.spec.js` cover these behaviors.
  - Sticky `.topbar` with computed background alpha `>= 0.85`
  - Mobile `390×844` autocomplete anchored with `0px` overlap below `View outlook` and `0px` horizontal overflow
  - Keyboard-only range switching and modal Escape/focus trapping. Users can open the price table manually during normal operation. If the chart chunk fails, the table opens automatically.
- **Cross-platform `package-lock.json`:** When updating npm dependencies on Windows, verify Linux optional dependencies (e.g., `@emnapi/runtime`) remain intact so `npm ci` passes on `ubuntu-24.04`.

---

## Git & Pull Request Workflow

- `main` is protected (`backend-and-research` and `frontend` checks required; no direct pushes or force pushes). Always create a feature branch and open a PR.
- Stage files explicitly (`git add <paths>`) — never use `git add -A` so local scratch files or untracked artifacts are not accidentally committed.
- On Windows PowerShell, pass multiline PR descriptions via `gh pr create --body-file <file>` / `gh pr edit --body-file <file>` to avoid backtick escape corruption.
