export function getApiBase() {
  return (import.meta.env.VITE_API_URL || (typeof window !== 'undefined' ? window.STOCKLSTM_API_BASE : '') || '').replace(/\/$/, '');
}

/** A deadline also settles transports which ignore AbortSignal. */
export async function withDeadline(operation, { signal, timeoutMs = 20_000 } = {}) {
  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
  const controller = new AbortController();
  let rejectLimit;
  const limit = new Promise((resolve, reject) => { rejectLimit = reject; });
  const cancel = () => { rejectLimit(new DOMException('Aborted', 'AbortError')); controller.abort(); };
  signal?.addEventListener('abort', cancel, { once: true });
  const timer = setTimeout(() => {
    rejectLimit(new DOMException('The request timed out. Please retry.', 'TimeoutError'));
    controller.abort();
  }, timeoutMs);
  try { return await Promise.race([operation(controller.signal), limit]); }
  finally { clearTimeout(timer); signal?.removeEventListener('abort', cancel); }
}

export async function requestJson(path, { signal, timeoutMs, baseUrl = getApiBase(), fetchImpl = (...args) => globalThis.fetch(...args) } = {}) {
  return withDeadline(async (requestSignal) => {
    const response = await fetchImpl(`${baseUrl.replace(/\/$/, '')}${path}`, { signal: requestSignal, cache: 'no-cache' });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof payload.detail === 'string' ? payload.detail : payload.message;
      const error = new Error(detail || `Request failed (${response.status}).`);
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload;
  }, { signal, timeoutMs });
}

export function sessionCacheKey(ticker) {
  const timezone = ticker.endsWith('.L') ? 'Europe/London' : 'America/New_York';
  const parts = new Intl.DateTimeFormat('en-GB', { timeZone: timezone, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).formatToParts(new Date());
  const values = Object.fromEntries(parts.map(({ type, value }) => [type, value]));
  const close = ticker.endsWith('.L') ? 16 * 60 + 30 : 16 * 60;
  const phase = Number(values.hour) * 60 + Number(values.minute) >= close ? 'after_close' : 'before_close';
  return `${getApiBase()}:${ticker}:${values.year}-${values.month}-${values.day}:${phase}`;
}

export function assertTickerIdentity(payload, ticker) {
  if (payload?.ticker !== ticker) throw new Error('Response ticker does not match the selected stock.');
  return payload;
}

export function assertMarketIdentity(payload, expected) {
  if (!expected) return payload;
  assertTickerIdentity(payload, expected.ticker);
  const asOf = payload.data_as_of || payload.evidence?.data_as_of || payload.as_of || payload.asOf;
  const expectedDate = expected.data_as_of || expected.asOf;
  if (asOf && expectedDate && asOf !== expectedDate) throw new Error('Market origins differ. Refresh the outlook.');
  const originPrice = payload.current_price;
  const expectedPrice = expected.current_price ?? expected.daily?.at(-1)?.c;
  const precision = expected.ticker?.endsWith('.L') ? 1 : 2;
  if (originPrice != null && expectedPrice != null && Number(originPrice).toFixed(precision) !== Number(expectedPrice).toFixed(precision)) throw new Error('Market origin prices differ. Refresh the outlook.');
  const fingerprint = payload.provenance?.data_fingerprint || payload.evidence?.data_fingerprint || payload.dataFingerprint;
  const expectedFingerprint = expected.provenance?.data_fingerprint || expected.dataFingerprint;
  if (fingerprint && expectedFingerprint && fingerprint !== expectedFingerprint) throw new Error('Market data identities differ. Refresh the outlook.');
  for (const [name, received, wanted] of [
    ['feed', payload.provenance?.data_feed || payload.evidence?.data_feed || payload.feed, expected.provenance?.data_feed || expected.feed],
    ['adjustment', payload.provenance?.price_adjustment || payload.evidence?.price_adjustment || payload.adjustment, expected.provenance?.price_adjustment || expected.adjustment],
  ]) {
    if (received && wanted && received !== 'unknown' && wanted !== 'unknown' && received !== wanted) throw new Error(`Market ${name} identities differ. Refresh the outlook.`);
  }
  return payload;
}


export function validatePriceResponse(payload, ticker) {
  assertTickerIdentity(payload, ticker);
  const positive = (v) => typeof v === 'number' && Number.isFinite(v) && v > 0;
  if (!positive(payload.current_price) || !/^\d{4}-\d{2}-\d{2}$/.test(payload.data_as_of || '') || !Array.isArray(payload.future_dates) || payload.future_dates.length !== 7) throw new Error('Price response has an invalid origin or horizon.');
  let previous = payload.data_as_of;
  for (const date of payload.future_dates) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || new Date(date).toISOString().slice(0, 10) !== date || date <= previous) throw new Error('Price response has invalid session dates.');
    previous = date;
  }
  if (payload.model?.status !== 'experimental_point_only') {
    if (![payload.lower_prices, payload.upper_prices].every((array) => Array.isArray(array) && array.length === 7 && array.every(positive))) throw new Error('Price response has invalid uncertainty bounds.');
    if (payload.lower_prices.some((value, index) => value > payload.upper_prices[index])) throw new Error('Price response has inverted uncertainty bounds.');
  }
  const hasPoint = Array.isArray(payload.predicted_prices) && payload.predicted_prices.length === 7 && payload.predicted_prices.every(positive);
  if (!hasPoint && payload.backtest) return { ...payload, backtest: null };
  return payload;
}
