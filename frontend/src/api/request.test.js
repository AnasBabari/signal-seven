import { expect, it, vi } from 'vitest';
import { withDeadline, assertMarketIdentity } from './request';

it('terminates a transport which ignores cancellation', async () => {
  vi.useFakeTimers();
  const result = withDeadline(() => new Promise(() => {}), { timeoutMs: 10 });
  const check = expect(result).rejects.toMatchObject({ name: 'TimeoutError' });
  await vi.advanceTimersByTimeAsync(10);
  await check;
  vi.useRealTimers();
});

it('rejects mismatched origin and fingerprint before evidence is displayed', () => {
  expect(() => assertMarketIdentity({ ticker: 'MSFT', data_as_of: '2026-09-02' }, { ticker: 'MSFT', asOf: '2026-09-03' })).toThrow('origins differ');
  expect(() => assertMarketIdentity({ ticker: 'MSFT', provenance: { data_fingerprint: 'a' } }, { ticker: 'MSFT', dataFingerprint: 'b' })).toThrow('identities differ');
});
