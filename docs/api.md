# API Reference

All endpoints are served by the FastAPI backend. The base URL in
production is the Render service root (for example
`https://stock-predictor-lstm.onrender.com`); that is the only value
the frontend needs in `VITE_API_URL`.

Every response is JSON. Arrays of daily bars are always ordered
oldest → newest. Unless stated otherwise, responses are cached
per-process for a few minutes (see
[architecture.md](architecture.md#warm-up-and-cache-strategy)).

> Response shapes below were captured from a running server against
> live market data. Full JSON examples are in
> [`docs/api-samples/`](api-samples/).

## Conventions

| aspect | detail |
|---|---|
| Ticker format | 1–12 chars, `A–Z`, `0–9`, `.`, `-`, `_`. LSE tickers use the `.L` suffix (`SHEL.L`). |
| Dates | ISO `YYYY-MM-DD` (daily), ISO 8601 with offset (intraday). |
| Times | `origin_date` / `as_of` refer to the last **completed** session, never an in-progress one. |
| Rate limiting | Per-IP, SlowAPI. Exceeded → `429` JSON. |
| Errors | `{"detail": "<human-readable>"}`, except where noted. |

## Health & readiness

### `GET /health`

Liveness probe. Always fast — it never touches market data or a model.

```json
{"status":"ok","version":"1.1.0","deployment":{"provider":"unknown","environment":"local","commit":null,"preview":false}}
```

### `GET /ready`

Readiness probe. Reports the upstream market-data dependency (circuit
breaker state, consecutive failures, last error) so you can tell
"not ready" from "upstream is down".

### `GET /models`

Reports the configured price candidates, volatility horizons and per-horizon auto policy,
training capacity and ledger configuration. A specific forecast's `evidence.fallback_used`
describes any fallback on that request.

### `GET /`

Root banner + version.

## Market data

### `GET /api/v1/history`

Price history for the chart. One round trip backs every range tab and
zoom level; the client derives 5D…MAX from the daily series.

| parameter | type | default | notes |
|---|---|---|---|
| `ticker` | string | `AAPL` | validated against the supported universe |
| — | — | — | rate limit 30/minute |

Response:

| field | type | notes |
|---|---|---|
| `ticker` | string | normalised, upper-case |
| `as_of` | string | last completed session |
| `provider` | string | which market-data provider answered |
| `market_data_cache` | string | `hit` / `miss`, useful when diagnosing latency |
| `first_date` | string \| null | oldest date in the returned series |
| `daily` | array of `{d, c}` | date + close, capped at the most recent **1500** sessions |
| `intraday` | array \| null | 24H bars; `null` until a live intraday source is configured |
| `intraday_session` | string \| null | |

### `GET /api/v1/intraday`

The 24H chart requests this route on demand for supported US equities. Inspect
`available` and `bars` before plotting. Daily history remains usable when intraday bars are unavailable.

### `GET /api/v1/search`

Symbol autocomplete.

| parameter | type | constraints |
|---|---|---|
| `query` | string | 1–100 chars |

Returns `{"results":[{"ticker","name","type"}, ...]}`. When the
upstream search is unreachable it degrades to an exact-symbol
fallback and adds `"degraded": true` rather than failing.

### `GET /api/v1/info`

Company profile.

| parameter | type | default |
|---|---|---|
| `ticker` | string | `AAPL` |

```json
{"ticker":"MSFT","name":"Microsoft Corporation","exchange":"NMS","currency":"USD",
 "marketCap":3650769387520,"peRatio":27.39,"fiftyTwoWeekHigh":553.72,
 "fiftyTwoWeekLow":349.2,"avgVolume":36277025,"dayHigh":494.39,"dayLow":489.8,
 "previousClose":493.95,"sector":"Technology","industry":"Software - Infrastructure"}
```

## Price forecast

### `GET /api/v1/forecast`

A learned seven-session price estimate. **Rate limited to 12/minute**
(it is the most expensive endpoint).

| parameter | type | default | notes |
|---|---|---|---|
| `ticker` | string | `MSFT` | must be in the supported universe |
| `days` | int | `7` | 1–7; the horizon is fixed at 7, other values are rejected with `400` |
| `model` | string | `auto` | `auto`, or a named candidate |

Response:

| field | type | notes |
|---|---|---|
| `ticker`, `ticker_name` | string | |
| `exchange_mic`, `exchange_name` | string | e.g. `XNAS` / `NASDAQ` |
| `currency`, `currency_symbol` | string | `USD` / `$`, `GBp` / `p` for LSE |
| `forecast_days` | int | always 7 |
| `data_as_of` | string | last completed session used |
| `current_price` | float | |
| `future_dates` | string[7] | next 7 **trading** days (exchange calendar aware) |
| `predicted_prices` | float[7] | central estimate |
| `lower_prices`, `upper_prices` | float[7] | per-session bounds when the requested model supplies intervals |
| `historical_dates`, `historical_prices` | arrays | recent actuals for context |
| `interval_calibration` | object | The UI labels the bounds “80% pointwise conformal range” when `method` is `volatility_scaled_online_split_conformal` and `m5_gate_verdict` is `PASS`. Other usable bounds receive the experimental label. |
| `model` | object | `{name, kind, feature_version, target, selection, candidate_validation_mae}` |
| `backtest` | object | see below |
| `provenance` | object | `{data_provider, market_data_cache, calendar, completed_daily_bars_only}` |
| `news_context` | object | cached headline sentiment, headline and SEC filing counts, availability and `used_in_point_model: false` |

`backtest` is a chronological holdout, not a fit:

```
split, test_start, test_end, test_samples, metric_source,
mae_percent, rmse_percent, direction_accuracy,
persistence_mae_percent, relative_mae_vs_persistence, per_horizon
```

`relative_mae_vs_persistence` is the honest headline: below `1.0`
means the model beat a naive "tomorrow = today" on that ticker.

## Volatility

Volatility endpoints estimate **how much** a price may move, never
which direction. See [methodology.md](methodology.md).

### `GET /api/v1/volatility/forecast`

| parameter | type | default | notes |
|---|---|---|---|
| `ticker` | string | `AAPL` | |
| `horizon` | int | `5` | one of **1, 5, 10, 20** sessions |
| `model` | string | `auto` | Default policy `v5_duan_har` selects `garch_11` at one session and `har_rv` at 5, 10 and 20. Request `gpu_g3` explicitly for the packaged ONNX model. |

Response (abridged):

| field | type | notes |
|---|---|---|
| `ticker`, `as_of`, `horizon` | | |
| `current_price` | float | |
| `historical_dates`, `historical_prices` | arrays | last 90 sessions |
| `forecast` | object | see below |
| `evidence` | object | ~35 keys of provenance (below) |

`forecast`:

```
future_dates, price_quantiles, expected_cumulative_variance_path,
expected_cumulative_variance, expected_annualized_volatility,
predicted_volatility, volatility_unit, model, requested_model, baseline
```

`evidence` carries the disclosure the UI shows under *How this
forecast works*: `model_status`, `model_family`, `model_name`,
`baseline` (true when it fell back), `model_version`,
`feature_set_version`, `model_policy_version`, `fallback_used`,
`test_evidence_qlike_vs_rolling` (null unless independently supplied), `risk_level`,
`risk_ratio_vs_trailing_60d`, `trailing_annualized_volatility_60d`.

When the requested learned model cannot be used, the response **says so**:
`baseline: true` plus a non-null `fallback_used`. It never silently
substitutes a different model.

The packaged G3 artifacts currently report `metric_source: validation_panel`.
That is historical validation evidence, not a claim of untouched production
test performance. A serving failure is disclosed with `model_status: baseline`
and a non-null `fallback_used`; the UI renders missing risk information as
`Unavailable`.

### `GET /api/v1/volatility/forecast/batch`

Example: `/api/v1/volatility/forecast/batch?ticker=MSFT&horizons=5,10,20&model=auto`.
The batch shares a market snapshot across requested horizons. `forecasts` and `errors`
are keyed by horizon, so callers can display successful results when one horizon fails.

### `GET /api/v1/volatility/ledger`

Immutable forward ledger of past forecasts, so live performance can
be measured later. Read-only.

### Authenticated volatility endpoints

These require a collector credential and are not part of the public
surface:

- `POST /api/v1/volatility/collect` — record one forecast.
- `GET /api/v1/volatility/export-ledger` — export the ledger.
- `POST /api/v1/volatility/score-ledger` — score recorded forecasts
  against realised outcomes.

## News

### `GET /api/v1/news`

Returns `items` and `sec_filings`. The SEC-only fallback also exposes its filings in
`items`. Duplicate filings are grouped by accession and excluded from headline sentiment.
News provides context for readers. Fetching the price forecast only reads the cached news summary.

## Error responses

| status | when |
|---|---|
| `400` | Unsupported parameter value, for example a 3-session volatility horizon |
| `404` | Ticker not in the supported universe / no market data |
| `422` | Market history too short or not finite enough to form a forecast |
| `429` | Rate limit exceeded |
| `503` | Upstream market data unavailable (circuit open), or the ledger is unavailable |

A `503` from `/api/v1/volatility/forecast` returns a structured body
rather than a bare `detail`, so callers can distinguish
`MARKET_DATA_UNAVAILABLE` from `FORECAST_LEDGER_UNAVAILABLE`.

## Server timing

`/api/v1/forecast` returns a `Server-Timing` header:

```
Server-Timing: data;dur=12, train_or_cache;dur=42, total;dur=55
```

`train_or_cache` is the useful one: a low value means the artifact
cache served the request, a high one means the model was actually
fitted on this request.
