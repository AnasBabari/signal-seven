import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  clearStockInfoCache,
  fetchStockInfo,
  formatCompactMarketCap,
  normalizeStockInfo,
} from './stockInfoClient';

describe('stockInfoClient', () => {
  beforeEach(() => {
    clearStockInfoCache();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('normalizes partial fundamentals with null P/E and caches for 60s', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        ticker: 'MSFT',
        name: 'Microsoft Corporation',
        sector: 'Information Technology',
        currency: 'USD',
        marketCap: 3_120_000_000_000,
        peRatio: null,
        fiftyTwoWeekLow: 385.5,
        fiftyTwoWeekHigh: 468.35,
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const first = await fetchStockInfo('msft');
    const second = await fetchStockInfo('MSFT');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(first).toEqual(second);
    expect(first.sector).toBe('Information Technology');
    expect(first.peRatio).toBeNull();
    expect(first.fiftyTwoWeekLow).toBe(385.5);
    expect(first.fiftyTwoWeekHigh).toBe(468.35);
    expect(first.hasAnyFundamental).toBe(true);
  });

  it('formats US and UK market caps in USD ($) and GBP (£) while honoring marketCapCurrency and USD-quoted LSE listings', () => {
    expect(formatCompactMarketCap(3_120_000_000_000, { currency: 'USD', currencySymbol: '$', ticker: 'MSFT' })).toBe('$3.12T');
    expect(formatCompactMarketCap(165_400_000_000, { currency: 'GBp', marketCapCurrency: 'GBP', currencySymbol: 'p', ticker: 'SHEL.L' })).toBe('£165.4B');
    expect(formatCompactMarketCap(45_200_000_000, { currency: 'USD', marketCapCurrency: 'USD', currencySymbol: '$', ticker: 'CPG.L' })).toBe('$45.2B');
    expect(formatCompactMarketCap(null, { currency: 'USD' })).toBe('—');
  });

  it('returns null gracefully on 404/503 without throwing', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({ detail: 'Upstream unavailable' }),
    }));
    expect(await fetchStockInfo('MSFT')).toBeNull();
  });

  it('drops inverted 52-week ranges safely', () => {
    const res = normalizeStockInfo({
      ticker: 'MSFT',
      sector: 'Technology',
      fiftyTwoWeekLow: 500,
      fiftyTwoWeekHigh: 400,
    }, 'MSFT');
    expect(res.fiftyTwoWeekLow).toBeNull();
    expect(res.fiftyTwoWeekHigh).toBeNull();
    expect(res.hasAnyFundamental).toBe(true);
  });
});
