import React from 'react';
import { expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import LazyLineChart from './LazyLineChart';

it('recovers from chunk failure with a retry while identifying the numerical fallback and notifying onErrorChange', async () => {
  const onErrorChange = vi.fn();
  const load = vi.fn()
    .mockRejectedValueOnce(new Error('Chunk unavailable'))
    .mockResolvedValueOnce({ Line: React.forwardRef(() => <canvas aria-label="Recovered chart" />) });
  render(<LazyLineChart loadEngine={load} onErrorChange={onErrorChange} />);
  expect(await screen.findByRole('alert')).toHaveTextContent('data table');
  expect(onErrorChange).toHaveBeenCalledWith(true);
  expect(screen.queryByText('Loading Chart Engine...')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Retry chart engine' }));
  expect(await screen.findByLabelText('Recovered chart')).toBeInTheDocument();
  expect(onErrorChange).toHaveBeenLastCalledWith(false);
});

it('terminates a never-resolving chart import with recovery controls', async () => {
  vi.useFakeTimers();
  render(<LazyLineChart loadEngine={() => new Promise(() => {})} />);
  await act(async () => { await vi.advanceTimersByTimeAsync(20_000); });
  expect(screen.getByRole('button', { name: /Retry chart engine/i })).toBeInTheDocument();
  vi.useRealTimers();
});
