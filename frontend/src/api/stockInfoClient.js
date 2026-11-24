import { requestJson, sessionCacheKey, assertTickerIdentity } from './request';

const cache = new Map();
const TTL_MS = 60_000;

export function clearStockInfoCache() {
  cache.clear();
}

function finitePositiveOrNull(value) {
  if (value == null || value === '' || typeof value === 'boolean') return null;
  const num = Number(value);
  return Number.isFinite(num) && num > 0 ? num : null;
}

function cleanTextOrNull(value) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (!trimmed || trimmed === '—' || trimmed === '-' || trimmed.toLowerCase() === 'unknown' || trimmed.toLowerCase() === 'n/a') {
    return null;
  }
  return trimmed;
}

export function formatCompactMarketCap(
  value,
  { currency = null, marketCapCurrency = null, currencySymbol = '$', ticker = '' } = {},
) {
  const num = finitePositiveOrNull(value);
  if (num == null) return '—';
  const cur = String(marketCapCurrency || currency || '').trim().toUpperCase();
  const sym = String(currencySymbol || '').trim();
  const isUkSymbol = String(ticker || '').trim().toUpperCase().endsWith('.L');
  // UK .L marketCap from market providers is denominated in GBP (£), even when
  // share prices are quoted in GBp (p), except USD-quoted LSE shares (CPG.L, IHG.L).
  let prefix = '$';
  if (cur === 'USD' || (!cur && sym === '$' && !isUkSymbol)) {
    prefix = '$';
  } else if (cur === 'GBP' || cur === 'GBX' || sym === 'p' || sym === 'GBp' || sym === '£' || isUkSymbol) {
    prefix = (cur === 'USD') ? '$' : '£';
  } else if (cur === 'EUR' || sym === '€') {
    prefix = '€';
  }

  if (num >= 1e12) return `${prefix}${(num / 1e12).toFixed(2)}T`;
  if (num >= 1e9) return `${prefix}${(num / 1e9).toFixed(1)}B`;
  if (num >= 1e6) return `${prefix}${(num / 1e6).toFixed(1)}M`;
  return `${prefix}${Math.round(num).toLocaleString('en-US')}`;
}

export function normalizeStockInfo(payload, symbol) {
  assertTickerIdentity(payload, symbol);
  const sector = cleanTextOrNull(payload.sector);
  const industry = cleanTextOrNull(payload.industry);
  const exchange = cleanTextOrNull(payload.exchange || payload.exchange_name);
  const currency = cleanTextOrNull(payload.currency) || (symbol.endsWith('.L') ? 'GBp' : 'USD');
  const marketCapCurrency = cleanTextOrNull(payload.marketCapCurrency ?? payload.market_cap_currency)
    || (currency === 'GBp' || currency === 'GBX' ? 'GBP' : currency);
  const currencySymbol = cleanTextOrNull(payload.currency_symbol)
    || (currency === 'GBp' || currency === 'GBX' ? 'p' : currency === 'GBP' ? '£' : '$');
  const marketCap = finitePositiveOrNull(payload.marketCap ?? payload.market_cap);
  const peRatio = finitePositiveOrNull(payload.peRatio ?? payload.pe_ratio ?? payload.trailingPE);
  const fiftyTwoWeekHigh = finitePositiveOrNull(payload.fiftyTwoWeekHigh ?? payload.fifty_two_week_high);
  const fiftyTwoWeekLow = finitePositiveOrNull(payload.fiftyTwoWeekLow ?? payload.fifty_two_week_low);
  const hasValid52WeekRange = fiftyTwoWeekLow != null
    && fiftyTwoWeekHigh != null
    && fiftyTwoWeekLow <= fiftyTwoWeekHigh;

  const hasAnyFundamental = Boolean(
    sector
    || marketCap != null
    || peRatio != null
    || hasValid52WeekRange,
  );

  return {
    ticker: symbol,
    name: cleanTextOrNull(payload.name) || symbol,
    exchange,
    currency,
    marketCapCurrency,
    currencySymbol,
    sector,
    industry,
    marketCap,
    peRatio,
    fiftyTwoWeekHigh: hasValid52WeekRange ? fiftyTwoWeekHigh : null,
    fiftyTwoWeekLow: hasValid52WeekRange ? fiftyTwoWeekLow : null,
    hasAnyFundamental,
  };
}

export async function fetchStockInfo(ticker, { signal, timeoutMs = 10_000, forceRefresh = false } = {}) {
  const symbol = String(ticker || '').trim().toUpperCase();
  if (!symbol) return null;
  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
  const key = `info:${sessionCacheKey(symbol)}`;
  const saved = cache.get(key);
  if (!forceRefresh && saved && Date.now() - saved.at < TTL_MS) {
    return saved.value;
  }
  try {
    const payload = await requestJson(`/api/v1/info?ticker=${encodeURIComponent(symbol)}`, {
      signal,
      timeoutMs,
      cache: forceRefresh ? 'no-cache' : undefined,
    });
    const value = normalizeStockInfo(payload, symbol);
    cache.set(key, { at: Date.now(), value });
    while (cache.size > 64) cache.delete(cache.keys().next().value);
    return value;
  } catch (error) {
    if (signal?.aborted || error?.name === 'AbortError') throw error;
    return null;
  }
}
