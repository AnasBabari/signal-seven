import { afterEach, expect, it, vi } from 'vitest';
import { fetchSimpleForecast, fetchTickerNews, wakeForecastService } from './simpleForecastClient';

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

it('surfaces a missing learned endpoint without inventing a price path or backtest', async () => {
  const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 404, json: async () => ({ detail: 'Not found' }) });
  vi.stubGlobal('fetch', fetchMock);
  await expect(fetchSimpleForecast('MSFT')).rejects.toMatchObject({ status: 404 });
  expect(fetchMock).toHaveBeenCalledTimes(1);
});

it('preserves an empty news response instead of substituting example headlines', async () => {
  const body = { ticker: 'MSFT', status: 'available', items: [] };
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => body }));
  expect(await fetchTickerNews('MSFT')).toEqual(body);
});

it('reports news failure as unavailable with no fabricated feed or timestamp', async () => {
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Offline')));
  expect(await fetchTickerNews('MSFT')).toMatchObject({ status: 'unavailable', items: [], provider: null, as_of: null });
});

it('deduplicates concurrent wakeForecastService callers and survives StrictMode unmount-remount', async () => {
  let resolveHealth;
  let internalSignal;
  const fetchMock = vi.fn((_url, init) => {
    internalSignal = init.signal;
    return new Promise((resolve) => {
      resolveHealth = resolve;
    });
  });
  vi.stubGlobal('fetch', fetchMock);

  const firstController = new AbortController();
  const secondController = new AbortController();
  const firstAttempts = [];
  const secondAttempts = [];

  // Simulate React 18 StrictMode: mount -> synchronous unmount -> synchronous remount
  const firstPromise = wakeForecastService({ signal: firstController.signal, onAttempt: (n) => firstAttempts.push(n) });
  firstController.abort();
  const secondPromise = wakeForecastService({ signal: secondController.signal, onAttempt: (n) => secondAttempts.push(n) });

  await expect(firstPromise).rejects.toMatchObject({ name: 'AbortError' });
  expect(internalSignal.aborted).toBe(false);
  expect(fetchMock).toHaveBeenCalledTimes(1);
  expect(secondAttempts).toEqual([1]);

  resolveHealth({ ok: true, json: async () => ({ status: 'ok' }) });
  expect(await secondPromise).toEqual({ status: 'ok' });
  expect(fetchMock).toHaveBeenCalledTimes(1);
});

it('aborts the shared wake loop once all subscribers genuinely unmount', async () => {
  let internalSignal;
  const fetchMock = vi.fn((_url, init) => {
    internalSignal = init.signal;
    return new Promise((_resolve, reject) => {
      init.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    });
  });
  vi.stubGlobal('fetch', fetchMock);

  const controller = new AbortController();
  const wakePromise = wakeForecastService({ signal: controller.signal });
  controller.abort();
  await expect(wakePromise).rejects.toMatchObject({ name: 'AbortError' });
  await Promise.resolve();
  expect(internalSignal.aborted).toBe(true);
});
