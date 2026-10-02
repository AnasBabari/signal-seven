import { getApiBase, requestJson, assertTickerIdentity, validatePriceResponse } from './request';
const API_BASE = getApiBase();
const getJson = requestJson;

export async function wakeForecastService({ signal, onAttempt } = {}) {
  let lastError;
  for (let attempt = 1; attempt <= 12; attempt += 1) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    onAttempt?.(attempt);
    try {
      const health = await getJson('/health', { signal, timeoutMs: 12_000 });
      if (health?.status === 'ok') return health;
    } catch (error) {
      if (error?.name === 'AbortError' && signal?.aborted) throw error;
      lastError = error;
    }
    await new Promise((resolve, reject) => {
      const cleanResolve = () => { signal?.removeEventListener('abort', onAbort); resolve(); };
      const delay = setTimeout(cleanResolve, Math.min(3_000 + attempt * 750, 10_000));
      const onAbort = () => {
        clearTimeout(delay);
        signal?.removeEventListener('abort', onAbort);
        reject(new DOMException('Aborted', 'AbortError'));
      };
      signal?.addEventListener('abort', onAbort, { once: true });
    });
  }
  throw lastError || new Error('The forecast service did not start in time.');
}

/** Request a learned forecast; unavailable service responses stay unavailable. */
export async function fetchSimpleForecast(ticker, { signal } = {}) {
  const symbol = String(ticker || '').trim().toUpperCase();
  if (!symbol) throw new Error('A stock ticker is required.');
  // Only the learned endpoint can supply a price estimate. A 404 is surfaced
  // to the UI, which retains history and offers a retry.
  return validatePriceResponse(await getJson(`/api/v1/forecast?ticker=${encodeURIComponent(symbol)}&days=7`, { signal, timeoutMs: 120_000 }), symbol);
}

export async function fetchTickerNews(ticker, { signal } = {}) {
  const symbol = String(ticker || '').trim().toUpperCase();
  try {
    const res = await getJson(`/api/v1/news?ticker=${encodeURIComponent(symbol)}`, {
      signal,
      timeoutMs: 15_000,
    });
    if (Array.isArray(res?.items)) {
      return assertTickerIdentity(res, symbol);
    }
  } catch (error) {
    if (signal?.aborted) throw error;
  }
  return {
    status: 'unavailable',
    ticker: symbol,
    items: [],
    role: 'context_only',
    used_by_model: false,
    provider: null,
    as_of: null,
  };
}

export { API_BASE };
