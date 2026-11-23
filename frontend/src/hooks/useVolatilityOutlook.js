import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchVolatilityForecast, fetchVolatilityForecastBatch } from '../ml/volatilityClient';
import { withDeadline, sessionCacheKey, assertMarketIdentity } from '../api/request';

export const OUTLOOK_HORIZONS = [5, 10, 20];
const TOTAL_TIMEOUT_MS = 20_000;
const outlookCache = new Map();
export function clearVolatilityOutlookCache() { outlookCache.clear(); }

export function useVolatilityOutlook(ticker, { refreshToken = 0, expectedIdentity = null } = {}) {
  const [outlook, setOutlook] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [retryTick, setRetryTick] = useState(0);
  const seqRef = useRef(0);
  const previousRefresh = useRef(refreshToken);
  const previousRetryTick = useRef(retryTick);
  const expectedRef = useRef(expectedIdentity);
  expectedRef.current = expectedIdentity;
  const expectedDate = expectedIdentity?.data_as_of || expectedIdentity?.asOf || '';
  const expectedFingerprint = expectedIdentity?.provenance?.data_fingerprint || expectedIdentity?.dataFingerprint || '';

  useEffect(() => {
    const symbol = String(ticker || '').trim().toUpperCase();
    const id = ++seqRef.current;
    const globalRefresh = previousRefresh.current !== refreshToken;
    const explicitRetry = previousRetryTick.current !== retryTick;
    previousRefresh.current = refreshToken;
    previousRetryTick.current = retryTick;

    if (!symbol) {
      setOutlook(null);
      setLoading(false);
      setError('');
      return undefined;
    }

    const key = `${sessionCacheKey(symbol)}:${expectedDate}:${expectedFingerprint}`;
    const cached = outlookCache.get(key);
    const fresh = cached && Date.now() - cached.at < 60_000;
    const byHorizon = fresh && !globalRefresh ? { ...cached.value.byHorizon } : {};
    const horizons = OUTLOOK_HORIZONS.filter((horizon) => !byHorizon[horizon]);
    const cacheMode = (globalRefresh || explicitRetry) ? 'no-cache' : undefined;
    const controller = new AbortController();
    let pending = horizons.length;

    const publish = () => {
      if (seqRef.current !== id || controller.signal.aborted) return;
      const hasSuccess = Object.keys(byHorizon).length > 0;
      const value = {
        ticker: symbol,
        byHorizon: { ...byHorizon },
        pendingHorizons: horizons.filter((h) => !byHorizon[h]),
      };
      setOutlook(hasSuccess ? value : null);
      setLoading(pending > 0);
      setError(pending === 0 && !hasSuccess ? 'Volatility outlook is unavailable right now.' : '');
      if (hasSuccess) {
        outlookCache.set(key, { at: Date.now(), value });
        while (outlookCache.size > 64) outlookCache.delete(outlookCache.keys().next().value);
      }
    };

    publish();
    if (horizons.length === 0) return undefined;

    const deadlineAt = Date.now() + TOTAL_TIMEOUT_MS;
    const remainingMs = () => Math.max(0, deadlineAt - Date.now());

    const runSingleHorizonFallback = (targetHorizons) => {
      const budget = remainingMs();
      if (budget <= 0 || controller.signal.aborted) {
        pending = 0;
        publish();
        return;
      }
      targetHorizons.forEach((horizon) => {
        withDeadline(
          (signal) => fetchVolatilityForecast(symbol, horizon, signal, { model: 'auto', timeoutMs: budget, cache: cacheMode }),
          { signal: controller.signal, timeoutMs: budget },
        )
          .then((result) => {
            assertMarketIdentity(result, expectedRef.current);
            byHorizon[horizon] = result;
          })
          .catch(() => {})
          .finally(() => {
            pending = Math.max(0, pending - 1);
            publish();
          });
      });
    };

    const canUseBatch = horizons.length > 1 && typeof fetchVolatilityForecastBatch === 'function';
    if (!canUseBatch) {
      runSingleHorizonFallback(horizons);
    } else {
      const batchBudget = remainingMs();
      withDeadline(
        (signal) => fetchVolatilityForecastBatch(symbol, horizons, signal, {
          model: 'auto',
          timeoutMs: batchBudget,
          cache: cacheMode,
        }),
        { signal: controller.signal, timeoutMs: batchBudget },
      )
        .then((batchResult) => {
          if (seqRef.current !== id || controller.signal.aborted) return;
          for (const horizon of horizons) {
            const entry = batchResult?.byHorizon?.[horizon];
            if (!entry) continue;
            try {
              assertMarketIdentity(entry, expectedRef.current);
              byHorizon[horizon] = entry;
            } catch {
              // Ignore horizon with mismatched market identity.
            }
          }
          pending = 0;
          publish();
        })
        .catch((err) => {
          if (seqRef.current !== id || controller.signal.aborted) return;
          if (err?.name === 'AbortError' || err?.name === 'TimeoutError' || remainingMs() <= 0) {
            pending = 0;
            publish();
            return;
          }
          runSingleHorizonFallback(horizons);
        });
    }

    return () => controller.abort();
  }, [ticker, retryTick, refreshToken, expectedDate, expectedFingerprint]);

  const retry = useCallback(() => setRetryTick((tick) => tick + 1), []);
  return { outlook, loading, error, retry };
}
