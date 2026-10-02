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
