import React from 'react';
import { expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import LazyLineChart from './LazyLineChart';

it('recovers from chunk failure with a retry while identifying the numerical fallback', async () => {
  const load = vi.fn().mockRejectedValueOnce(new Error('Chunk unavailable')).mockResolvedValueOnce([
    { Chart: { register: vi.fn() } }, { Line: React.forwardRef(() => <canvas aria-label="Recovered chart" />) },
  ]);
  render(<LazyLineChart loadEngine={load} />);
  expect(await screen.findByRole('alert')).toHaveTextContent('data table');
  expect(screen.queryByText('Loading Chart Engine...')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Retry chart engine' }));
  expect(await screen.findByLabelText('Recovered chart')).toBeInTheDocument();
});

it('terminates a never-resolving chart import with recovery controls', async () => {
  vi.useFakeTimers();
  render(<LazyLineChart loadEngine={() => new Promise(() => {})} />);
  await act(async () => { await vi.advanceTimersByTimeAsync(20_000); });
  expect(screen.getByRole('button', { name: /Retry chart engine/i })).toBeInTheDocument();
  vi.useRealTimers();
});
