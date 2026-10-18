# Learned forecast artifact reuse

`/api/v1/forecast` keeps its existing learned selection, fitting, clipping and response semantics. Set `FORECAST_MODEL_CACHE_DIR` to a private directory to enable fitted artifacts; unset means memory-only caching.

For sklearn candidates, artifacts include the fitted pipeline (including its scaler), final inference inputs, ordered feature schema, target clipping bounds, validation residual bounds, training row count, data cutoff, cache identity, and original evaluation/response metadata. On a cold load the service deserializes the model and performs prediction; it does not rebuild features, select candidates or train. It verifies exact agreement with the saved prices before returning the response.

Explicit GPU point inference requires a configured, content-hashed, self-describing checkpoint with verified preprocessing, ordered features, architecture and role. Auto does not use legacy deployment checkpoints. This experimental point-only path supplies no historical metrics or bounds and does not create a sklearn artifact. Fresh-process equivalence below applies to sklearn artifacts.

Identity includes ticker, last observation, fingerprint of the full frame, selected model request, candidate configuration, feature/cache implementation versions, implementation bytes, runtime library versions and checkpoint content hashes. Changes invalidate reuse. No old response-only JSON cache is used by the new artifact path.

Writes publish a checksummed content-addressed blob before atomically replacing its manifest. Missing, oversized, corrupt, incompatible or non-equivalent artifacts cause a cache miss and normal learned training. No baseline substitution is introduced. Disk errors do not prevent serving the freshly trained response.

Security: joblib is pickle-based and **must only read service-owned files**. Never accept uploaded artifacts or use a directory writable by untrusted users. SHA256 detects accidental corruption; it is not authentication against an attacker able to replace both files.

Operational limits: files survive process restarts only while their filesystem survives. Render ephemeral storage is not cross-deploy/cross-instance persistence. Shared durable artifact storage and automatic old-blob retention remain separate infrastructure choices. Default fitting capacity is one per worker (`FORECAST_TRAINING_MAX_CONCURRENCY`, range 1–4). Same-ticker waiting is bounded to two seconds, then returns 503 with Retry-After 3. Memory hits bypass fitting locks. Optional `FORECAST_TRAINING_COORDINATION_DATABASE_URL` uses a bounded PostgreSQL session advisory lock to serialize fitting across workers; absent configuration means process-only coordination. No database rows or schemas are changed. A coordination outage fails as retryable busy, with no prediction substitution.

Responses include this invocation's data, artifact lookup, model deserialization, features, training, inference and total timings plus cache status. Cached prediction/evaluation metadata stays exact; timings are refreshed rather than copied from an earlier request.

Verification: cold-process serialized-response equivalence, fitted scaler/prediction round-trip, key/runtime/version mismatch, corrupt/missing blob, path traversal rejection, and the existing session/revised-data invalidation and concurrent-request tests. No production values or UI layouts are changed by this storage layer.
