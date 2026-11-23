import { act, renderHook } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { clearVolatilityOutlookCache, useVolatilityOutlook } from './useVolatilityOutlook';
import { fetchVolatilityForecast, fetchVolatilityForecastBatch } from '../ml/volatilityClient';

vi.mock('../ml/volatilityClient', () => ({
  fetchVolatilityForecast: vi.fn(),
  fetchVolatilityForecastBatch: vi.fn(),
}));

afterEach(() => {
  vi.useRealTimers();
  clearVolatilityOutlookCache();
  vi.resetAllMocks();
});

it('uses the batch endpoint, handles partial horizon failures, and retries only missing horizons', async () => {
  fetchVolatilityForecastBatch.mockResolvedValueOnce({
    ticker: 'MSFT',
    byHorizon: {
      5: { ticker: 'MSFT', forecast: { predicted_volatility: 0.2 } },
      20: { ticker: 'MSFT', forecast: { predicted_volatility: 0.22 } },
    },
    errors: { 10: { code: 'HORIZON_UNAVAILABLE', message: 'unavailable' } },
  });
  const { result, rerender } = renderHook(
    ({ refreshToken }) => useVolatilityOutlook('MSFT', { refreshToken }),
    { initialProps: { refreshToken: 0 } },
  );
  await act(async () => { await Promise.resolve(); });
  expect(fetchVolatilityForecastBatch).toHaveBeenCalledTimes(1);
  expect(fetchVolatilityForecast).not.toHaveBeenCalled();
  expect(result.current.outlook.byHorizon[5]).toBeTruthy();
  expect(result.current.outlook.byHorizon[20]).toBeTruthy();
  expect(result.current.outlook.byHorizon[10]).toBeUndefined();
  expect(result.current.loading).toBe(false);

  // Retry missing period fetches only horizon 10 with no-cache.
  fetchVolatilityForecast.mockResolvedValueOnce({ ticker: 'MSFT', forecast: { predicted_volatility: 0.21 } });
  await act(async () => { result.current.retry(); });
  expect(fetchVolatilityForecast).toHaveBeenCalledTimes(1);
  expect(fetchVolatilityForecast.mock.calls[0][1]).toBe(10);
  expect(fetchVolatilityForecast.mock.calls[0][3]).toMatchObject({ cache: 'no-cache' });
  expect(result.current.outlook.byHorizon[10]).toBeTruthy();

  // Global refresh increments refreshToken -> forces fresh batch fetch with no-cache.
  fetchVolatilityForecastBatch.mockResolvedValueOnce({
    ticker: 'MSFT',
    byHorizon: {
      5: { ticker: 'MSFT' },
      10: { ticker: 'MSFT' },
      20: { ticker: 'MSFT' },
    },
    errors: {},
  });
  await act(async () => { rerender({ refreshToken: 1 }); });
  expect(fetchVolatilityForecastBatch).toHaveBeenCalledTimes(2);
  expect(fetchVolatilityForecastBatch.mock.calls[1][3]).toMatchObject({ cache: 'no-cache' });
});

it('falls back to bounded parallel single-horizon requests (<=20s total) when batch returns 404', async () => {
  vi.useFakeTimers();
  fetchVolatilityForecastBatch.mockRejectedValue(Object.assign(new Error('Not Found'), { httpStatus: 404 }));
  fetchVolatilityForecast.mockImplementation((ticker, horizon) => (
    horizon === 5 ? Promise.resolve({ ticker, forecast: { predicted_volatility: 0.2 } }) : new Promise(() => {})
  ));
  const { result } = renderHook(() => useVolatilityOutlook('MSFT', { refreshToken: 0 }));
  await act(async () => { await Promise.resolve(); await Promise.resolve(); });
  expect(result.current.outlook.byHorizon[5]).toBeTruthy();
  expect(result.current.loading).toBe(true);
  await act(async () => { await vi.advanceTimersByTimeAsync(20_000); });
  expect(result.current.loading).toBe(false);
  expect(result.current.outlook.byHorizon[5]).toBeTruthy();
  expect(fetchVolatilityForecast.mock.calls.map((call) => call[1])).toEqual([5, 10, 20]);
});

it('preserves 60s cache on remount even when refreshToken > 0 has not changed', async () => {
  fetchVolatilityForecastBatch.mockResolvedValueOnce({
    ticker: 'MSFT',
    byHorizon: {
      5: { ticker: 'MSFT' },
      10: { ticker: 'MSFT' },
      20: { ticker: 'MSFT' },
    },
    errors: {},
  });
  const first = renderHook(() => useVolatilityOutlook('MSFT', { refreshToken: 2 }));
  await act(async () => { await Promise.resolve(); });
  expect(fetchVolatilityForecastBatch).toHaveBeenCalledTimes(1);
  first.unmount();

  const second = renderHook(() => useVolatilityOutlook('MSFT', { refreshToken: 2 }));
  await act(async () => { await Promise.resolve(); });
  expect(fetchVolatilityForecastBatch).toHaveBeenCalledTimes(1);
  expect(second.result.current.outlook.byHorizon[5]).toBeTruthy();
  expect(second.result.current.loading).toBe(false);
});

it('does not extend total timeout past 20s when batch request hangs', async () => {
  vi.useFakeTimers();
  fetchVolatilityForecastBatch.mockImplementation(() => new Promise(() => {}));
  const { result } = renderHook(() => useVolatilityOutlook('MSFT', { refreshToken: 0 }));
  await act(async () => { await vi.advanceTimersByTimeAsync(20_000); });
  expect(result.current.loading).toBe(false);
  expect(result.current.error).toMatch(/unavailable/i);
  expect(fetchVolatilityForecast).not.toHaveBeenCalled();
});
