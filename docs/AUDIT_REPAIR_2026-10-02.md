# Audit repair verification — 2 October 2026

## Ledger compatibility

New live records use `deployable_v5:annualized_sigma_v1` as the immutable ledger model version. Predicted, recent and realized volatility are annualized sigma fractions. Recent volatility is `sqrt(252 * rolling_c2c_20[H1])`.

Existing rows and fingerprints are preserved. Legacy direction metrics are invalidated at read time because their recent-volatility units are unverified; magnitude errors remain usable. Reports expose the number of direction-eligible scored rows. No automatic rewrite or production migration is performed. Any future correction requires a separately preserved, explicitly versioned derivation from the original market snapshot, with original rows retained.

Settlement uses canonical daily data, scores only mature rows, separates total/pending/scored counts and preserves live/replay tracks. Collector retries bounded 429 responses using Retry-After and supports an explicit local resume manifest. No live collection, settlement, export or deployment was run during this repair.

## Price evidence

The primary display remains the range midpoint. Point forecast metrics describe `predicted_prices`; midpoint and error-band coverage are unavailable unless separately evaluated. Reused chronological windows are retrospective. Legacy GPU deployment checkpoints cannot enter historical model selection or supply metrics without verified pre-evaluation training and scaler provenance.
