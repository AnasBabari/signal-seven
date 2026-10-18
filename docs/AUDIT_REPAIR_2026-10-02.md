# Audit repair verification — 2 October 2026

## Ledger compatibility

New live records use `deployable_v5:annualized_sigma_v1` as the immutable ledger model version. Predicted, recent and realized volatility are annualized sigma fractions. Recent volatility is `sqrt(252 * rolling_c2c_20[H1])`.

Existing rows and fingerprints are preserved. Legacy direction metrics are invalidated at read time because their recent-volatility units are unverified; magnitude errors remain usable. Reports expose the number of direction-eligible scored rows. No automatic rewrite or production migration is performed. Any future correction requires a separately preserved, explicitly versioned derivation from the original market snapshot, with original rows retained.

Settlement uses canonical daily data, scores only mature rows, separates total/pending/scored counts and preserves live/replay tracks. Collector retries bounded 429 responses using Retry-After and supports an explicit local resume manifest. No live collection, settlement, export or deployment was run during this repair.

## Price evidence

The primary display remains the range midpoint. Point forecast metrics describe `predicted_prices`; midpoint and error-band coverage are unavailable unless separately evaluated. Reused chronological windows are retrospective. Legacy GPU deployment checkpoints cannot enter historical model selection or supply metrics without verified pre-evaluation training and scaler provenance.

## Market-data contract

History and forecasting use the same service: configured US chain, and explicit Yahoo daily bars for LSE. UK routing is exchange support, not a silent US feed fallback. No provider equivalence or paid feed requirement is asserted. Feed, adjustment, completed origin and full daily-data fingerprint travel with responses; live ledger provider provenance includes feed/adjustment/fingerprint when available. Cache identity includes feed, adjustment, years and contract version. Old cache namespaces are not reused.

Session labels retain their provider-local calendar date; instant timestamps are explicitly converted to the exchange timezone. Impossible OHLC, missing/invalid values, nonpositive prices and negative volume fail closed. Zero-range sessions use neutral close-location 0.5 in both training and serving; latest feature origin must equal the declared origin. G3 reads immutable tuple observations captured by the snapshot and cannot redownload across a rollover.

Stage 2 focused verification: 105 backend contract tests passed, including BST, partial UK session, zero-range final bar, provider cache identity and G3 no-second-read checks.

## Price evidence and pooled chronology

Pooled GPU partitions use union-calendar cutoffs shared by all tickers; a training label must end before every validation origin, and a validation label before every test origin. Preprocessing remains train-only. Selection restoration and best-epoch schedule-prefix refits are retained. Selection checkpoints are saved before any evaluation, with ordered features, architecture/config, roles, row/scaler hashes, target cutoffs, dataset protocol and a file SHA-256 sidecar. Historical checkpoint inference rejects missing provenance or overlapping training labels. Explicit unsupported price model requests fail; auto remains the fixed Ridge/random-forest policy. Verified explicitly configured GPU point predictions are experimental with no attached historical evaluation or bounds.

The active price response now exposes requested/selected models, train-majority direction comparison and retrospective per-horizon error-band coverage, width and both tail misses. These measurements describe the historical evaluation model. Coverage of the current all-data refit remains unavailable. News stays context_only; news archive features and coverage use exchange-specific causal close cutoffs, including half-days, and retain the existing revision gate.

Fixed local development comparison completed on cached AAPL/ARM/BP.L with Ridge alpha100, a small two-epoch LSTM, persistence and train-majority direction. No test evaluation or deployment refit was run. Artifacts: `artifacts/audit_price_comparison_20261002/` contains protocol, source hashes/resolved paths, checkpoints, predictions/losses, HAC/Holm date-aggregated paired results and SHA256SUMS. This is a bounded implementation check using reused historical data, not independent scientific evidence.

## Recovery and accessible inspection

All data clients share a request deadline which settles even transports ignoring cancellation. Ticker and known market identity mismatches fail closed. Browser caches are bounded to 60 seconds and keyed by API base, exchange date/close phase and available identity; global retries force history refresh, while horizon retries preserve successful sibling results. Volatility horizons publish independently and always leave loading after their deadline. Price and history failures remain actionable even if neither produces a chart. Lazy chart imports offer retry and a numerical fallback.

The chart keeps its visual interaction and adds full session dates, keyboard date inspection, an expandable data table, accessible historical/midpoint descriptions and pressed range controls. The primary experimental midpoint summary appears above the chart. Company-name autocomplete uses the active universe. Reachability and recorded forecasts use narrower labels. Statistical volatility baselines expose the same trailing risk context as the learned path.

## Capacity, cache and release verification

Price fitting defaults to one job per worker across tickers. Same-ticker stampede waiting is bounded to two seconds; occupied capacity or unavailable coordination returns 503 with Retry-After 3 and no forecast substitution. Memory hits bypass fitting locks. Explicit optional PostgreSQL session advisory locking coordinates workers; local and ephemeral caches do not. No live settings, credential files, paid services or deployment manifests were changed. Auto warmup no longer imports unused PyTorch.

Fitted sklearn artifacts retain runtime/version/data/schema/checksum guards and exact fresh-process prediction equivalence. Request timings separate data, artifact lookup, deserialization, features, training/evaluation, final inference and total duration, with fresh cache status. Model discovery reports the actual 286-ticker universe, point-only retrospective metrics and unavailable midpoint/refit coverage. CI now runs the focused local browser contract suite and preserves failure evidence.

Production GET verification is preserved in `reports/AUDIT_PRODUCTION_2026-10-02.json`. The intended Vercel project/domain and backend serve `dc66d50` on main, before these repairs. Frontend routing points to `https://stock-predictor-lstm.onrender.com`. Initial backend cold checks timed out; one bounded warm retry returned health/ready 200, market-data completed session 2026-10-01, and durable PostgreSQL ledger available. Warm US history, price and volatility returned 200 at origin 2026-10-01. UK price/volatility returned 404; deployed UK history returned an in-progress 2026-10-02 bar at 10:30 UTC. This exposes the existing deployed origin/routing mismatch that the local canonical service fixes; these local changes have not been published.

The Render connector requires a user-confirmed workspace before service-detail inspection. No workspace was selected or mutated. Actual Render service name/region remain unverified; render.yaml declares stock-predictor-backend/oregon. This is recorded drift to reconcile before a separately authorized deployment, not silently rewritten infrastructure.

Local browser verification uses fixtures, not live market predictions. Screenshots are `artifacts/audit-ui-desktop.png` and `artifacts/audit-ui-mobile.png`. Chromium checks include empty first visit, pressed range controls, keyboard date inspection/table, modal focus containment, supporting-tab navigation, company autocomplete, rapid stock switching, invalid origins/bounds, partial and never-resolving horizons, chunk failure recovery, touch scrolling and mobile overflow. The standalone agent-browser native session could not connect; Playwright supplied the passing browser checks and reviewed images.

No push, deployment, authenticated collection, settlement, live-ledger export or production correction was performed. Existing data/options user files remain untouched. New compatibility labels leave legacy ledger records immutable. The bounded two-epoch development price comparison does not beat persistence or the train-majority direction comparator and does not establish independent predictive superiority. Legacy checkpoints and missing archival outputs were not repurposed as evidence.

Final acceptance: 300 backend/research tests, 163 frontend tests and 14 Chromium browser tests pass. Ruff check and format check pass; Vite production build passes. One upstream Starlette/httpx deprecation warning remains. The GPU comparison and visual evidence are preserved as small, explicitly scoped local artifacts; temporary test/runtime directories are removed.

## Authorized release review

Following explicit authorization to review, push and verify production, the Render control plane
confirmed existing service `srv-d9i97s7avr4c73adg050`, named `stock-predictor-lstm`, in
`frankfurt` on the free plan. It tracks `main` with `checksPass` automatic deployments and
previews disabled. The blueprint is aligned to those observed settings; no service relocation,
replacement, creation, secret update or live-ledger operation is required. Vercel project
`prj_xVDiEtgqOpmZKeyzNEPC9tNSOOe4` is the existing `stock-predictor-lstm` project.

The remote README-only commit `dc66d50` was preserved by a normal merge before release.
The repair tree independently passed the same 300 Python and 163 frontend tests, lint,
formatting and production build checks. All 14 fixture-based browser contracts also passed;
desktop/mobile fixture captures were refreshed. The blueprint passed validation against Render's
official JSON schema. Post-push CI/deployment probes remain separate release checks; the earlier
production report above describes the pre-release deployment, not evidence that these changes
were already live.

The first pushed run (`37014217914`) passed the frontend job but stopped at backend Ruff
import ordering: Linux classified the repository-owned `scripts` namespace differently.
Backend isort policy now declares that namespace first-party explicitly; the test is retained
and no forecasting or ledger behavior is changed by this CI correction.
