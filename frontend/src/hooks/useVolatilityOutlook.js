import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchVolatilityForecast } from '../ml/volatilityClient';
import { withDeadline, sessionCacheKey, assertMarketIdentity } from '../api/request';
export const OUTLOOK_HORIZONS = [5, 10, 20];
const outlookCache = new Map();
export function clearVolatilityOutlookCache() { outlookCache.clear(); }

export function useVolatilityOutlook(ticker, { refreshToken = 0, expectedIdentity = null } = {}) {
  const [outlook, setOutlook] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [retryTick, setRetryTick] = useState(0);
  const seqRef = useRef(0);
  const previousRefresh = useRef(refreshToken);
  const expectedRef = useRef(expectedIdentity);
  expectedRef.current = expectedIdentity;
  const expectedDate = expectedIdentity?.data_as_of || expectedIdentity?.asOf || '';
  const expectedFingerprint = expectedIdentity?.provenance?.data_fingerprint || expectedIdentity?.dataFingerprint || '';
  useEffect(() => {
    const symbol = String(ticker || '').trim().toUpperCase();
    const id = ++seqRef.current;
    if (!symbol) { setOutlook(null); setLoading(false); setError(''); return undefined; }
    const key = `${sessionCacheKey(symbol)}:${expectedDate}:${expectedFingerprint}`;
    const globalRefresh = previousRefresh.current !== refreshToken;
    previousRefresh.current = refreshToken;
    const cached = outlookCache.get(key);
    const fresh = cached && Date.now() - cached.at < 60_000;
    const byHorizon = fresh && !globalRefresh && (!refreshToken || retryTick > 0) ? { ...cached.value.byHorizon } : {};
    const horizons = OUTLOOK_HORIZONS.filter((horizon) => !byHorizon[horizon]);
    const controller = new AbortController();
    let pending = horizons.length;
    const publish = () => {
      if (seqRef.current !== id || controller.signal.aborted) return;
      const hasSuccess = Object.keys(byHorizon).length > 0;
      const value = { ticker: symbol, byHorizon: { ...byHorizon }, pendingHorizons: horizons.filter((h) => !byHorizon[h]) };
      setOutlook(hasSuccess ? value : null);
      setLoading(pending > 0);
      setError(pending === 0 && !hasSuccess ? 'Volatility outlook is unavailable right now.' : '');
      if (hasSuccess) {
        outlookCache.set(key, { at: Date.now(), value });
        while (outlookCache.size > 64) outlookCache.delete(outlookCache.keys().next().value);
      }
    };
    publish();
    horizons.forEach((horizon) => {
      withDeadline((signal) => fetchVolatilityForecast(symbol, horizon, signal, { model: 'auto' }), { signal: controller.signal, timeoutMs: 20_000 })
        .then((result) => { assertMarketIdentity(result, expectedRef.current); byHorizon[horizon] = result; })
        .catch(() => {})
        .finally(() => { pending -= 1; publish(); });
    });
    return () => controller.abort();
  }, [ticker, retryTick, refreshToken, expectedDate, expectedFingerprint]);
  const retry = useCallback(() => setRetryTick((tick) => tick + 1), []);
  return { outlook, loading, error, retry };
}
