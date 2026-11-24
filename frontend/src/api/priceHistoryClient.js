import { requestJson, sessionCacheKey, assertTickerIdentity } from './request';
import { hasValidIntradayBars } from '../utils/priceRanges';
const cache = new Map();
const intradayCache = new Map();
const TTL_MS = 60_000;
export function clearPriceHistoryCache() {
  cache.clear();
  intradayCache.clear();
}

export async function fetchPriceHistory(ticker, { signal, timeoutMs = 20_000, forceRefresh = false } = {}) {
  const symbol = String(ticker || '').trim().toUpperCase();
  if (!symbol) throw new Error('A ticker symbol is required for price history.');
  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
  const key = sessionCacheKey(symbol);
  const saved = cache.get(key);
  if (!forceRefresh && saved && Date.now() - saved.at < TTL_MS) return { ...saved.value, meta: { fetchMs: 0, fromCache: true } };
  const started = performance.now();
  const payload = assertTickerIdentity(
    await requestJson(`/api/v1/history?ticker=${encodeURIComponent(symbol)}`, {
      signal,
      timeoutMs,
      cache: forceRefresh ? 'no-cache' : undefined,
    }),
    symbol,
  );
  const daily = payload?.daily;
  if (!Array.isArray(daily) || !daily.length) throw new Error('No price history is available for this ticker.');
  let previous = '';
  for (const bar of daily) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(bar.d) || !Number.isFinite(Date.parse(bar.d)) || bar.d <= previous || typeof bar.c !== 'number' || !Number.isFinite(bar.c) || bar.c <= 0) throw new Error('Price history has invalid dates or values.');
    previous = bar.d;
  }
  if (payload.as_of && payload.as_of !== previous) throw new Error('Price history origin does not match its final session.');
  const inlineIntraday = Array.isArray(payload.intraday) ? payload.intraday : null;
  const value = {
    ticker: symbol,
    asOf: payload.as_of || previous,
    provider: payload.provider || null,
    feed: payload.feed,
    adjustment: payload.adjustment,
    dataFingerprint: payload.data_fingerprint,
    marketDataCache: payload.market_data_cache || null,
    firstDate: payload.first_date || daily[0].d,
    currencySymbol: payload.currency_symbol || null,
    daily,
    intraday: inlineIntraday,
    intradayAvailable: Boolean(payload.intraday_available || hasValidIntradayBars(inlineIntraday)),
    intradaySession: payload.intraday_session || null,
  };
  cache.set(key, { at: Date.now(), value });
  while (cache.size > 64) cache.delete(cache.keys().next().value);
  return { ...value, meta: { fetchMs: Math.round(performance.now() - started), fromCache: false } };
}

export async function fetchIntradayHistory(ticker, { signal, timeoutMs = 15_000, forceRefresh = false } = {}) {
  const symbol = String(ticker || '').trim().toUpperCase();
  if (!symbol) throw new Error('A ticker symbol is required for intraday history.');
  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
  const key = `intraday:${sessionCacheKey(symbol)}`;
  const saved = intradayCache.get(key);
  if (!forceRefresh && saved && Date.now() - saved.at < TTL_MS) {
    return saved.value;
  }
  const payload = assertTickerIdentity(
    await requestJson(`/api/v1/intraday?ticker=${encodeURIComponent(symbol)}`, {
      signal,
      timeoutMs,
      cache: forceRefresh ? 'no-cache' : undefined,
    }),
    symbol,
  );
  const rawBars = Array.isArray(payload?.intraday)
    ? payload.intraday
    : (Array.isArray(payload?.bars) ? payload.bars : []);
  const bars = [];
  let prevTs = -Infinity;
  for (const bar of rawBars) {
    const ts = Date.parse(bar?.t);
    const close = Number(bar?.c);
    if (!bar?.t || !Number.isFinite(ts) || ts <= prevTs || !Number.isFinite(close) || close <= 0) {
      throw new Error('Intraday 24H bars contain invalid timestamps or prices.');
    }
    prevTs = ts;
    bars.push({ t: String(bar.t), c: close });
  }
  if (!hasValidIntradayBars(bars)) {
    throw new Error('Not enough intraday bars for 24H view (minimum 5 required).');
  }
  const value = {
    ticker: symbol,
    bars,
    intradaySession: payload.intraday_session || payload.session || payload.as_of || null,
  };
  intradayCache.set(key, { at: Date.now(), value });
  while (intradayCache.size > 64) intradayCache.delete(intradayCache.keys().next().value);
  return value;
}
