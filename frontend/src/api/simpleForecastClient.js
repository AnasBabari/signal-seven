import { getApiBase, requestJson, assertTickerIdentity, validatePriceResponse } from './request';
const API_BASE = getApiBase();
const getJson = requestJson;

let activeWake = null;

function createSharedWake() {
  const controller = new AbortController();
  const wakeState = {
    controller,
    attempt: 0,
    subscribers: new Set(),
    promise: null,
  };

  const notifyAttempt = (attempt) => {
    wakeState.attempt = attempt;
    wakeState.subscribers.forEach((subscriber) => {
      try {
        subscriber.onAttempt?.(attempt);
      } catch {
        // Subscriber callback errors must not break the shared wake loop.
      }
    });
  };

  wakeState.promise = (async () => {
    let lastError;
    const { signal } = controller;
    for (let attempt = 1; attempt <= 12; attempt += 1) {
      if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
      notifyAttempt(attempt);
      try {
        const health = await getJson('/health', { signal, timeoutMs: 12_000, cache: 'no-cache' });
        if (health?.status === 'ok') return health;
      } catch (error) {
        if (error?.name === 'AbortError' && signal.aborted) throw error;
        lastError = error;
      }
      await new Promise((resolve, reject) => {
        if (signal.aborted) {
          reject(new DOMException('Aborted', 'AbortError'));
          return;
        }
        const onAbort = () => {
          clearTimeout(delay);
          signal.removeEventListener('abort', onAbort);
          reject(new DOMException('Aborted', 'AbortError'));
        };
        const cleanResolve = () => {
          signal.removeEventListener('abort', onAbort);
          resolve();
        };
        const delay = setTimeout(cleanResolve, Math.min(3_000 + attempt * 750, 10_000));
        signal.addEventListener('abort', onAbort, { once: true });
      });
    }
    throw lastError || new Error('The forecast service did not start in time.');
  })().finally(() => {
    if (activeWake === wakeState) {
      activeWake = null;
    }
    wakeState.subscribers.clear();
  });

  // Prevent unhandled rejection if all subscribers unsubscribe before the internal loop rejects.
  wakeState.promise.catch(() => {});

  return wakeState;
}

export function wakeForecastService({ signal, onAttempt } = {}) {
  if (signal?.aborted) {
    return Promise.reject(new DOMException('Aborted', 'AbortError'));
  }

  if (!activeWake || activeWake.controller.signal.aborted) {
    activeWake = createSharedWake();
  }
  const wakeState = activeWake;
  const subscriber = { onAttempt };
  wakeState.subscribers.add(subscriber);
  if (wakeState.attempt > 0) {
    onAttempt?.(wakeState.attempt);
  }

  return new Promise((resolve, reject) => {
    let settled = false;

    const cleanupSubscriber = () => {
      wakeState.subscribers.delete(subscriber);
      signal?.removeEventListener('abort', onCallerAbort);
      // Defer aborting the shared internal loop to a microtask so React 18
      // StrictMode mount -> unmount -> remount keeps the shared wake loop alive.
      if (wakeState.subscribers.size === 0) {
        queueMicrotask(() => {
          if (activeWake === wakeState && wakeState.subscribers.size === 0) {
            wakeState.controller.abort();
            activeWake = null;
          }
        });
      }
    };

    const onCallerAbort = () => {
      if (settled) return;
      settled = true;
      cleanupSubscriber();
      reject(new DOMException('Aborted', 'AbortError'));
    };

    signal?.addEventListener('abort', onCallerAbort, { once: true });

    wakeState.promise.then(
      (health) => {
        if (settled) return;
        settled = true;
        signal?.removeEventListener('abort', onCallerAbort);
        wakeState.subscribers.delete(subscriber);
        resolve(health);
      },
      (error) => {
        if (settled) return;
        settled = true;
        signal?.removeEventListener('abort', onCallerAbort);
        wakeState.subscribers.delete(subscriber);
        reject(error);
      },
    );
  });
}

/** Request a learned forecast; unavailable service responses stay unavailable. */
export async function fetchSimpleForecast(ticker, { signal, forceRefresh = false } = {}) {
  const symbol = String(ticker || '').trim().toUpperCase();
  if (!symbol) throw new Error('A stock ticker is required.');
  // Only the learned endpoint can supply a price estimate. A 404 is surfaced
  // to the UI, which retains history and offers a retry.
  return validatePriceResponse(
    await getJson(`/api/v1/forecast?ticker=${encodeURIComponent(symbol)}&days=7`, {
      signal,
      timeoutMs: 120_000,
      cache: forceRefresh ? 'no-cache' : undefined,
    }),
    symbol,
  );
}

export async function fetchTickerNews(ticker, { signal, forceRefresh = false } = {}) {
  const symbol = String(ticker || '').trim().toUpperCase();
  try {
    const res = await getJson(`/api/v1/news?ticker=${encodeURIComponent(symbol)}`, {
      signal,
      timeoutMs: 15_000,
      cache: forceRefresh ? 'no-cache' : undefined,
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
