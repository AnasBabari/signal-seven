import { requestJson, sessionCacheKey, assertTickerIdentity } from './request';
const cache = new Map();
const TTL_MS = 60_000;
export function clearPriceHistoryCache() { cache.clear(); }

export async function fetchPriceHistory(ticker, { signal, timeoutMs = 20_000, forceRefresh = false } = {}) {
  const symbol = String(ticker || '').trim().toUpperCase();
  if (!symbol) throw new Error('A ticker symbol is required for price history.');
  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
  const key = sessionCacheKey(symbol);
  const saved = cache.get(key);
  if (!forceRefresh && saved && Date.now() - saved.at < TTL_MS) return { ...saved.value, meta: { fetchMs: 0, fromCache: true } };
  const started = performance.now();
  const payload = assertTickerIdentity(await requestJson(`/api/v1/history?ticker=${encodeURIComponent(symbol)}`, { signal, timeoutMs }), symbol);
  const daily = payload?.daily;
  if (!Array.isArray(daily) || !daily.length) throw new Error('No price history is available for this ticker.');
  let previous = '';
  for (const bar of daily) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(bar.d) || !Number.isFinite(Date.parse(bar.d)) || bar.d <= previous || typeof bar.c !== 'number' || !Number.isFinite(bar.c) || bar.c <= 0) throw new Error('Price history has invalid dates or values.');
    previous = bar.d;
  }
  if (payload.as_of && payload.as_of !== previous) throw new Error('Price history origin does not match its final session.');
  const value = { ticker: symbol, asOf: payload.as_of || previous, provider: payload.provider || null,
    feed: payload.feed, adjustment: payload.adjustment, dataFingerprint: payload.data_fingerprint,
    marketDataCache: payload.market_data_cache || null, firstDate: payload.first_date || daily[0].d,
    daily, intraday: Array.isArray(payload.intraday) ? payload.intraday : null, intradaySession: payload.intraday_session || null };
  cache.set(key, { at: Date.now(), value });
  while (cache.size > 64) cache.delete(cache.keys().next().value);
  return { ...value, meta: { fetchMs: Math.round(performance.now() - started), fromCache: false } };
}
