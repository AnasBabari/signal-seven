import { act, renderHook } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { clearVolatilityOutlookCache, useVolatilityOutlook } from './useVolatilityOutlook';
import { fetchVolatilityForecast } from '../ml/volatilityClient';
vi.mock('../ml/volatilityClient', () => ({ fetchVolatilityForecast: vi.fn() }));
afterEach(() => { vi.useRealTimers(); clearVolatilityOutlookCache(); vi.resetAllMocks(); });

it('shows success immediately, terminates a never-resolving sibling and retries only missing horizons', async () => {
  vi.useFakeTimers();
  fetchVolatilityForecast.mockImplementation((ticker, horizon) => horizon === 5 ? Promise.resolve({ ticker, forecast: { predicted_volatility: .2 } }) : new Promise(() => {}));
  const { result } = renderHook(() => useVolatilityOutlook('MSFT'));
  await act(async () => { await Promise.resolve(); });
  expect(result.current.outlook.byHorizon[5]).toBeTruthy();
  expect(result.current.loading).toBe(true);
  await act(async () => { await vi.advanceTimersByTimeAsync(20_000); });
  expect(result.current.loading).toBe(false);
  expect(result.current.outlook.byHorizon[5]).toBeTruthy();
  fetchVolatilityForecast.mockImplementation((ticker) => Promise.resolve({ ticker }));
  await act(async () => { result.current.retry(); });
  expect(fetchVolatilityForecast.mock.calls.map((call) => call[1])).toEqual([5, 10, 20, 10, 20]);
  expect(result.current.outlook.byHorizon[20]).toBeTruthy();
});
