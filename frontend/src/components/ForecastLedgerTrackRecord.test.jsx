import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import ForecastLedgerTrackRecord from './ForecastLedgerTrackRecord';
import { clearForecastLedgerCache } from '../api/forecastLedgerClient';

const mockLedgerResponse = {
  ticker: 'AAPL',
  horizon: 5,
  live_track_record: {
    record_source: 'live',
    total_forecasts: 2,
    scored_forecasts: 1,
    mean_mae: 0.014,
    mean_rmse: 0.014,
    mean_qlike: 0.5123,
    direction_accuracy_pct: 100.0,
  },
  replay_track_record: {
    record_source: 'historical_replay',
    total_forecasts: 25,
    scored_forecasts: 25,
    mean_mae: 0.0245,
    mean_rmse: 0.0312,
    mean_qlike: 0.7206,
    direction_accuracy_pct: 68.4,
  },
  track_record: {
    record_source: 'live',
    total_forecasts: 2,
    scored_forecasts: 1,
    mean_mae: 0.014,
    mean_rmse: 0.014,
    mean_qlike: 0.5123,
    direction_accuracy_pct: 100.0,
  },
  entries: [
    {
      id: 1,
      forecast_date: '2026-08-20',
      ticker: 'AAPL',
      horizon: 5,
      target_date: '2026-08-27',
      model_name: 'rolling_mean',
      predicted_volatility: 0.245,
      actual_realized_volatility: 0.231,
      forecast_error: 0.014,
      abs_error: 0.014,
      qlike_loss: 0.5123,
      status: 'scored',
      record_source: 'live',
    },
    {
      id: 2,
      forecast_date: '2026-08-25',
      ticker: 'AAPL',
      horizon: 5,
      target_date: '2026-09-01',
      model_name: 'rolling_mean',
      predicted_volatility: 0.252,
      actual_realized_volatility: null,
      forecast_error: null,
      abs_error: null,
      qlike_loss: null,
      status: 'pending',
      record_source: 'live',
    },
    {
      id: 3,
      forecast_date: '2026-07-15',
      ticker: 'AAPL',
      horizon: 5,
      target_date: '2026-07-22',
      model_name: 'rolling_mean',
      predicted_volatility: 0.22,
      actual_realized_volatility: 0.21,
      forecast_error: 0.01,
      abs_error: 0.01,
      qlike_loss: 0.6,
      status: 'scored',
      record_source: 'historical_replay',
    },
  ],
};

describe('ForecastLedgerTrackRecord', () => {
  beforeEach(() => {
    clearForecastLedgerCache();
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockLedgerResponse),
      })
    ));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders track record metrics and forecast entries without verified wording once loaded', async () => {
    const { container } = render(<ForecastLedgerTrackRecord ticker="AAPL" horizon={5} defaultOpen={true} enabled={true} />);

    // Hidden (null) while loading to prevent empty-ledger flashes
    expect(container.firstChild).toBeNull();

    expect(await screen.findByText('Recorded volatility forecasts')).toBeInTheDocument();
    expect(screen.queryByText(/verified/i)).not.toBeInTheDocument();
    expect(screen.getByText('AAPL 5 trading sessions')).toBeInTheDocument();
    expect(screen.getAllByText('5 trading sessions').length).toBeGreaterThan(0);

    // Headline live settlements KPI shows 1 scored live forecast
    expect(screen.getByText('1')).toBeInTheDocument();
    expect(screen.getByText('1.40%')).toBeInTheDocument();
    expect(screen.getAllByText('0.5123').length).toBeGreaterThanOrEqual(1);

    // Source chips
    expect(screen.getAllByText('Live').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Historical test').length).toBeGreaterThan(0);

    // Filter by replay tab
    const replayTab = screen.getByRole('button', { name: /Historical tests \(25\)/i });
    fireEvent.click(replayTab);

    // Under replay tab, replay metrics are shown
    expect(screen.getByText('25')).toBeInTheDocument();
    expect(screen.getByText('2.45%')).toBeInTheDocument();
    expect(screen.getByText('0.7206')).toBeInTheDocument();
  });

  it('returns null without fetching when enabled is false, and latches per ticker once enabled', async () => {
    const { container, rerender } = render(<ForecastLedgerTrackRecord ticker="AAPL" horizon={5} enabled={false} />);
    expect(container.firstChild).toBeNull();
    expect(global.fetch).not.toHaveBeenCalled();

    // Enabling fetches once and renders card
    rerender(<ForecastLedgerTrackRecord ticker="AAPL" horizon={5} enabled={true} />);
    expect(await screen.findByText('Recorded volatility forecasts')).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1);

    // Disabling keeps same ticker latched (no refetch)
    rerender(<ForecastLedgerTrackRecord ticker="AAPL" horizon={5} enabled={false} />);
    expect(screen.getByText('Recorded volatility forecasts')).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1);

    // Switching ticker while disabled resets latch (no eager fetch)
    rerender(<ForecastLedgerTrackRecord ticker="MSFT" horizon={5} enabled={false} />);
    expect(container.firstChild).toBeNull();
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it('returns null when there are no ledger records or when the request errors', async () => {
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ entries: [], live_track_record: { total_forecasts: 0 }, replay_track_record: { total_forecasts: 0 } }),
      })
    ));

    const { container, unmount } = render(<ForecastLedgerTrackRecord ticker="AAPL" horizon={5} defaultOpen={true} enabled={true} />);
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
    expect(container.firstChild).toBeNull();
    unmount();

    clearForecastLedgerCache();
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('Ledger down'))));
    const errRender = render(<ForecastLedgerTrackRecord ticker="AAPL" horizon={5} defaultOpen={true} enabled={true} />);
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
    expect(errRender.container.firstChild).toBeNull();
  });

  it('renders null when ticker is omitted', () => {
    const { container } = render(<ForecastLedgerTrackRecord ticker="" horizon={5} />);
    expect(container.firstChild).toBeNull();
  });

  it('cancels an old ticker request and ignores its late response', async () => {
    let finishOld;
    let oldSignal;
    vi.stubGlobal('fetch', vi.fn()
      .mockImplementationOnce((_url, { signal }) => {
        oldSignal = signal;
        return new Promise((resolve) => { finishOld = resolve; });
      })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ entries: [] }) }));
    const { container, rerender } = render(<ForecastLedgerTrackRecord ticker="AAPL" horizon={5} defaultOpen enabled />);
    rerender(<ForecastLedgerTrackRecord ticker="MSFT" horizon={20} defaultOpen enabled />);
    expect(oldSignal.aborted).toBe(true);
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2));
    await act(async () => {
      finishOld({ ok: true, json: async () => mockLedgerResponse });
    });
    expect(container.firstChild).toBeNull();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });
});
