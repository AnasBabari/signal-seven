import React from 'react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import PriceChart from './PriceChart';

vi.mock('../hooks/usePriceHistory', () => ({
  usePriceHistory: vi.fn(),
}));

vi.mock('../api/priceHistoryClient', () => ({
  fetchIntradayHistory: vi.fn(),
}));

vi.mock('../hooks/useVolatilityOutlook', () => ({
  useVolatilityOutlook: vi.fn(() => ({ outlook: null, loading: false, error: '', retry: vi.fn() })),
}));

import { usePriceHistory } from '../hooks/usePriceHistory';
import { fetchIntradayHistory } from '../api/priceHistoryClient';
import { useVolatilityOutlook } from '../hooks/useVolatilityOutlook';

let lastChartProps = null;
vi.mock('./LazyLineChart', () => ({
  __esModule: true,
  default: React.forwardRef((props, ref) => {
    lastChartProps = props;
    React.useImperativeHandle(ref, () => ({
      chartArea: { left: 0, right: 400, top: 0, bottom: 300, width: 400 },
      scales: {
        x: { getValueForPixel: (px) => (px / 400) * 199 },
        y: { getPixelForValue: () => 150 },
      },
      config: { options: {} },
      update: () => {},
    }));
    return <canvas data-testid="mock-chart" />;
  }),
}));

function historyFixture(sessions = 300, withIntraday = true) {
  return {
    ticker: 'MSFT',
    daily: Array.from({ length: sessions }, (_, i) => ({
      d: new Date(Date.UTC(2024, 5, 1 - (sessions - 1 - i))).toISOString().slice(0, 10),
      c: 100 + i,
    })),
    intraday: withIntraday
      ? Array.from({ length: 20 }, (_, i) => ({ t: `2024-06-01T14:${String(i).padStart(2, '0')}:00+00:00`, c: 200 + i }))
      : null,
  };
}

const forecastFixture = {
  ticker: 'MSFT',
  data_as_of: '2024-06-01',
  current_price: 399,
  future_dates: ['2024-06-03', '2024-06-04', '2024-06-05', '2024-06-06', '2024-06-07', '2024-06-10', '2024-06-11'],
  predicted_prices: [401, 402, 403, 404, 405, 406, 407],
  historical_error_band: {
    upper_prices: [405, 406, 407, 408, 409, 410, 411],
    lower_prices: [397, 398, 399, 400, 401, 402, 403],
  },
};

beforeEach(() => {
  vi.clearAllMocks();
  lastChartProps = null;
});

describe('PriceChart', () => {
  it('defaults to 1M and hides 5Y for a 2024 IPO', () => {
    usePriceHistory.mockReturnValue({ history: historyFixture(500, true), loading: false, error: '', retry: vi.fn() });
    render(<PriceChart ticker="MSFT" currencySymbol="$" />);
    expect(screen.getByRole('button', { name: '1M' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.queryByRole('button', { name: '5Y' })).toBeNull();
    expect(screen.getByRole('button', { name: '1Y' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'MAX' })).toBeInTheDocument();
  });

  it('hides 24H when intraday bars failed to load', () => {
    usePriceHistory.mockReturnValue({ history: historyFixture(3000, false), loading: false, error: '', retry: vi.fn() });
    render(<PriceChart ticker="MSFT" currencySymbol="$" />);
    expect(screen.queryByRole('button', { name: '24H' })).toBeNull();
    expect(screen.getByRole('button', { name: '5Y' })).toBeInTheDocument();
  });

  it('shows header price and period change', () => {
    usePriceHistory.mockReturnValue({ history: historyFixture(300, true), loading: false, error: '', retry: vi.fn() });
    render(<PriceChart ticker="MSFT" currencySymbol="$" />);
    expect(screen.getByText('MSFT')).toBeInTheDocument();
    expect(screen.getByText(/\+5\.56%/)).toBeInTheDocument();
  });

  it('switches ranges and overlays a clean solid silver continuation line without a canvas cone while keeping range in tooltip and table', () => {
    usePriceHistory.mockReturnValue({ history: historyFixture(300, true), loading: false, error: '', retry: vi.fn() });
    render(<PriceChart ticker="MSFT" currencySymbol="$" forecast={forecastFixture} />);
    fireEvent.click(screen.getByRole('button', { name: '1M' }));
    expect(screen.getByRole('button', { name: '1M' })).toHaveAttribute('aria-pressed', 'true');
    const roles = lastChartProps.data.datasets.map((ds) => ds.role);
    expect(roles).toEqual(['history', 'midpoint']);
    const labels = lastChartProps.data.datasets.map((ds) => ds.label);
    expect(labels).toEqual(['Price', 'Range midpoint — experimental']);
    expect(labels.join(' ')).not.toMatch(/calibrated|empirical band/i);
    expect(lastChartProps.data.labels.length).toBe(22 + 7);
    // Solid neutral silver continuation line anchored cleanly at index N-1 (21) with last close (399).
    const midpoint = lastChartProps.data.datasets.find((ds) => ds.role === 'midpoint');
    expect(midpoint.borderDash).toEqual([]);
    expect(midpoint.borderColor).toBe('#9ca3af');
    expect(midpoint.fill).toBe(false);
    expect(midpoint.data.slice(0, 21).every((v) => v == null)).toBe(true);
    expect(midpoint.data[21]).toBe(399);
    expect(lastChartProps.data.forecastSplitIndex).toBe(21);
    expect(lastChartProps.plugins.map((p) => p.id)).toContain('t212ForecastRegion');

    // Tooltip filter excludes N-1 anchored midpoint duplicate
    const { filter, callbacks } = lastChartProps.options.plugins.tooltip;
    expect(filter({ dataset: midpoint, dataIndex: 21 })).toBe(false);
    expect(filter({ dataset: midpoint, dataIndex: 22 })).toBe(true);

    // Midpoint tooltip still computes % relative to latest close (399) and afterLabel renders Range
    const midLabel = callbacks.label({ dataset: midpoint, parsed: { y: 401 }, dataIndex: 22 });
    expect(midLabel).toContain('$401.00');
    expect(midLabel).toContain('+0.50%');
    expect(callbacks.afterLabel({ dataset: midpoint, dataIndex: 22 })).toBe(' Range: $397.00 – $405.00');
    expect(callbacks.afterLabel({ dataset: lastChartProps.data.datasets[0], dataIndex: 21 })).toBe('');
  });

  it('gates wheel zoom behind Ctrl/Cmd and supports compact zoom buttons and double-click reset', () => {
    usePriceHistory.mockReturnValue({ history: historyFixture(300, true), loading: false, error: '', retry: vi.fn() });
    const { container } = render(<PriceChart ticker="MSFT" currencySymbol="$" />);
    fireEvent.click(screen.getByRole('button', { name: '1M' }));
    expect(screen.queryByRole('button', { name: 'Reset' })).toBeNull();
    // Plain wheel does NOT hijack scrolling or zoom
    fireEvent.wheel(container.querySelector('.t212-chart-wrap'), { clientX: 200, deltaY: -100 });
    expect(screen.queryByRole('button', { name: 'Reset' })).toBeNull();
    // Ctrl + wheel DOES zoom and reveals Reset
    fireEvent.wheel(container.querySelector('.t212-chart-wrap'), { clientX: 200, deltaY: -100, ctrlKey: true });
    expect(screen.getByRole('button', { name: 'Reset' })).toBeInTheDocument();
    expect(lastChartProps.options.animation).toBe(false);
    fireEvent.doubleClick(container.querySelector('.t212-chart-wrap'));
    expect(screen.queryByRole('button', { name: 'Reset' })).toBeNull();
    // Compact + / - zoom buttons work without mouse wheel
    fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
    expect(screen.getByRole('button', { name: 'Reset' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Zoom out' }));
    expect(screen.queryByRole('button', { name: 'Reset' })).toBeNull();
  });
  it('shows loading and error states', () => {
    const onSettled = vi.fn();
    usePriceHistory.mockReturnValue({ history: null, loading: true, error: '', meta: null, retry: vi.fn() });
    const { unmount } = render(<PriceChart ticker="MSFT" currencySymbol="$" onHistorySettled={onSettled} />);
    expect(screen.getByRole('status')).toHaveTextContent('Loading price history');
    expect(onSettled).not.toHaveBeenCalled();
    unmount();
    const onRetry = vi.fn();
    usePriceHistory.mockReturnValue({ history: null, loading: false, error: 'Down', meta: null, retry: onRetry });
    render(<PriceChart ticker="MSFT" currencySymbol="$" onHistorySettled={onSettled} />);
    fireEvent.click(screen.getByRole('button', { name: 'Retry chart' }));
    expect(onRetry).toHaveBeenCalled();
    expect(onSettled).toHaveBeenCalledWith(expect.objectContaining({ ok: false }));
  });

  it('reports settlement timing once per ticker', () => {
    const onSettled = vi.fn();
    usePriceHistory.mockReturnValue({
      history: historyFixture(300, true), loading: false, error: '',
      meta: { fetchMs: 412, fromCache: false }, retry: vi.fn(),
    });
    const { rerender } = render(<PriceChart ticker="MSFT" currencySymbol="$" onHistorySettled={onSettled} />);
    expect(onSettled).toHaveBeenCalledTimes(1);
    expect(onSettled).toHaveBeenCalledWith(expect.objectContaining({ ok: true, fetchMs: 412, fromCache: false }));
    rerender(<PriceChart ticker="MSFT" currencySymbol="$" onHistorySettled={onSettled} />);
    expect(onSettled).toHaveBeenCalledTimes(1);
  });

  it('falls back to forecast-embedded history when the history endpoint fails', () => {
    usePriceHistory.mockReturnValue({ history: null, loading: false, error: 'Down', meta: null, retry: vi.fn() });
    const fallbackForecast = {
      ...forecastFixture,
      historical_dates: ['2024-05-30', '2024-06-01'],
      historical_prices: [398, 399],
    };
    render(<PriceChart ticker="MSFT" currencySymbol="$" forecast={fallbackForecast} />);
    expect(screen.queryByRole('button', { name: 'Retry chart' })).toBeNull();
    expect(screen.getByRole('button', { name: 'MAX' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.queryByRole('button', { name: '5D' })).toBeNull();
    const labels = lastChartProps.data.datasets.map((ds) => ds.label);
    expect(labels).toContain('Range midpoint — experimental');
  });

  it('renders clean continuation line without shaded cone datasets or unverified calibration claims', () => {
    usePriceHistory.mockReturnValue({ history: historyFixture(300, true), loading: false, error: '', meta: null, retry: vi.fn() });
    const { container } = render(<PriceChart ticker="MSFT" currencySymbol="$" forecast={forecastFixture} />);
    const roles = lastChartProps.data.datasets.map((ds) => ds.role);
    expect(roles).toEqual(['history', 'midpoint']);
    expect(container.querySelector('.chart-estimate-bar')).toHaveTextContent('Historical error range (experimental)');
    expect(container.querySelector('.chart-estimate-bar').textContent).not.toMatch(/calibrated|empirical band/i);
  });

  it('refuses manufactured history instead of charting it', () => {
    usePriceHistory.mockReturnValue({ history: null, loading: false, error: 'Down', meta: null, retry: vi.fn() });
    render(
      <PriceChart
        ticker="MSFT"
        currencySymbol="$"
        forecast={{
          ...forecastFixture,
          historical_dates: ['2024-05-30', '2024-05-31'],
          historical_prices: [398, 399],
          historical_provenance: 'synthetic',
        }}
      />,
    );
    expect(screen.getByRole('button', { name: 'Retry chart' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'MAX' })).toBeNull();
    expect(screen.queryByTestId('mock-chart')).toBeNull();
  });

  it('never shows placeholder zeros while history is loading', () => {
    usePriceHistory.mockReturnValue({ history: null, loading: true, error: '', meta: null, retry: vi.fn() });
    const { container } = render(<PriceChart ticker="MSFT" currencySymbol="$" />);
    // The ticker stays visible, but an unknown price must read as unknown
    // rather than as a real "$0.00 (+0.00%)".
    expect(screen.getByText('MSFT')).toBeInTheDocument();
    expect(screen.queryByText(/\$0\.00/)).toBeNull();
    expect(screen.queryByText(/\(0\.00%\)/)).toBeNull();
    expect(container.querySelector('.t212-price-skeleton')).not.toBeNull();
  });

  it('puts the sign outside the currency symbol for a falling period', () => {
    const falling = {
      ticker: 'MSFT',
      daily: Array.from({ length: 300 }, (_, i) => ({
        d: new Date(Date.UTC(2024, 5, 1 - (299 - i))).toISOString().slice(0, 10),
        c: 400 - i,
      })),
      intraday: null,
    };
    usePriceHistory.mockReturnValue({ history: falling, loading: false, error: '', retry: vi.fn() });
    render(<PriceChart ticker="MSFT" currencySymbol="$" />);
    // "-$1.00" reads correctly; "$-1.00" does not.
    expect(screen.getByText(/\(-\d+\.\d{2}%\)/)).toBeInTheDocument();
    expect(screen.queryByText(/\$-/)).toBeNull();
  });

  it('renders a skeleton chart instead of a blank loading region', () => {
    usePriceHistory.mockReturnValue({ history: null, loading: true, error: '', meta: null, retry: vi.fn() });
    const { container } = render(<PriceChart ticker="MSFT" currencySymbol="$" />);
    expect(screen.getByText('Loading price history…')).toBeInTheDocument();
    expect(container.querySelector('.t212-chart-skeleton')).not.toBeNull();
    expect(container.querySelectorAll('.t212-skeleton-bars i').length).toBeGreaterThan(0);
    expect(screen.queryByTestId('mock-chart')).toBeNull();
  });
});


it('offers an opt-in Table toolbar toggle and auto-opens the table on chart engine error', () => {
  usePriceHistory.mockReturnValue({ history: historyFixture(300, false), loading: false, error: '', retry: vi.fn() });
  render(<PriceChart ticker="MSFT" forecast={forecastFixture} />);
  expect(screen.queryByRole('slider', { name: 'Inspect chart date' })).toBeNull();
  const tableBtn = screen.getByRole('button', { name: 'Table' });
  expect(tableBtn).toHaveAttribute('aria-expanded', 'false');
  expect(screen.queryByRole('table')).toBeNull();

  // Clicking Table opens the accessible table
  fireEvent.click(tableBtn);
  expect(tableBtn).toHaveAttribute('aria-expanded', 'true');
  expect(screen.getByRole('table')).toBeInTheDocument();
  expect(screen.getByRole('rowheader', { name: '2024-06-01' })).toBeInTheDocument();
  expect(screen.getByRole('rowheader', { name: '2024-06-11' })).toBeInTheDocument();

  // Closing and triggering LazyLineChart chunk error auto-opens the table
  fireEvent.click(tableBtn);
  expect(screen.queryByRole('table')).toBeNull();
  act(() => {
    lastChartProps.onErrorChange(true);
  });
  expect(screen.getByRole('table')).toBeInTheDocument();

  // X-axis tick callback uses CategoryScale getLabelForValue and formats compactly
  const tickCallback = lastChartProps.options.scales.x.ticks.callback;
  const formatted = tickCallback.call({ getLabelForValue: (v) => lastChartProps.data.labels[v] }, 0);
  expect(formatted).toMatch(/^\d{2}-\d{2}$/);
  expect(lastChartProps['aria-label']).toContain('MSFT historical closes');
});

it('lazily fetches /api/v1/intraday when user clicks 24H and history has intraday_available: true', async () => {
  const lazyBars = Array.from({ length: 6 }, (_, i) => ({
    t: `2024-06-01T14:0${i}:00Z`,
    c: 400 + i,
  }));
  fetchIntradayHistory.mockResolvedValueOnce({ ticker: 'MSFT', bars: lazyBars, intradaySession: '2024-06-01' });
  const hist = { ...historyFixture(300, false), intradayAvailable: true };
  const onRangeChange = vi.fn();
  usePriceHistory.mockReturnValue({ history: hist, loading: false, error: '', retry: vi.fn() });

  render(<PriceChart ticker="MSFT" onRangeChange={onRangeChange} />);
  expect(fetchIntradayHistory).not.toHaveBeenCalled();
  const btn24h = screen.getByRole('button', { name: '24H' });
  fireEvent.click(btn24h);
  expect(onRangeChange).toHaveBeenCalledWith('24H');
  expect(fetchIntradayHistory).toHaveBeenCalledWith('MSFT', expect.anything());
  await act(async () => { await Promise.resolve(); });
  expect(btn24h).toHaveAttribute('aria-pressed', 'true');
  expect(lastChartProps.data.labels).toHaveLength(6);
});

it('renders Trading 212 bottom range bar below canvas with crisp zero-tension polyline, faint horizontal gridlines, and dotted anchor baseline', () => {
  usePriceHistory.mockReturnValue({ history: historyFixture(300, true), loading: false, error: '', retry: vi.fn() });
  const { container } = render(<PriceChart ticker="MSFT" currencySymbol="$" forecast={forecastFixture} />);

  const chartWrap = container.querySelector('.t212-chart-wrap');
  const bottomBar = container.querySelector('.t212-bottom-bar');
  expect(chartWrap).not.toBeNull();
  expect(bottomBar).not.toBeNull();
  // Bottom range bar sits strictly after chart canvas wrap in DOM order
  expect(chartWrap.compareDocumentPosition(bottomBar) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(bottomBar.querySelector('.t212-range-tabs')).not.toBeNull();
  expect(bottomBar.querySelector('.table-toggle-btn')).not.toBeNull();

  // Sharp zero-tension angular polyline matching reference display (no shaded cone datasets)
  const historyDs = lastChartProps.data.datasets.find((ds) => ds.role === 'history');
  const midpointDs = lastChartProps.data.datasets.find((ds) => ds.role === 'midpoint');
  expect(lastChartProps.data.datasets).toHaveLength(2);
  expect(historyDs.tension).toBe(0);
  expect(midpointDs.tension).toBe(0);
  expect(midpointDs.borderDash).toEqual([]);

  // Subtle horizontal reference gridlines + horizontal dotted anchor baseline with solid dot
  expect(lastChartProps.options.scales.y.grid.display).toBe(true);
  expect(lastChartProps.options.plugins.t212ForecastRegion.anchorValue).toBe(399);

  const regionPlugin = lastChartProps.plugins.find((p) => p.id === 't212ForecastRegion');
  const calls = [];
  const fakeCtx = {
    save: () => {},
    restore: () => {},
    beginPath: () => {},
    setLineDash: (d) => calls.push(['dash', d]),
    moveTo: (x, y) => calls.push(['moveTo', x, y]),
    lineTo: (x, y) => calls.push(['lineTo', x, y]),
    stroke: () => calls.push(['stroke']),
    arc: (x, y, r) => calls.push(['arc', x, y, r]),
    fill: () => calls.push(['fill']),
  };
  regionPlugin.afterDatasetsDraw({
    ctx: fakeCtx,
    chartArea: { left: 20, right: 420, top: 10, bottom: 260 },
    scales: {
      x: { getPixelForValue: () => 280 },
      y: { getPixelForValue: () => 120 },
    },
    data: lastChartProps.data,
    options: lastChartProps.options,
  });
  // Draws horizontal dotted baseline across chartArea.left..right at anchor y=120 (no vertical wall)
  expect(calls).toContainEqual(['moveTo', 20, 120]);
  expect(calls).toContainEqual(['lineTo', 420, 120]);
  expect(calls).not.toContainEqual(['moveTo', 280, 10]);
  // Draws solid circular anchor dot at (280, 120)
  expect(calls.some((c) => c[0] === 'arc' && c[1] === 280 && c[2] === 120 && c[3] >= 4)).toBe(true);
});

it('scrubs hero price, delta, date, and estimate range live on hover and resets on pointer leave', () => {
  usePriceHistory.mockReturnValue({ history: historyFixture(300, false), loading: false, error: '', retry: vi.fn() });
  const { container } = render(<PriceChart ticker="MSFT" currencySymbol="$" forecast={forecastFixture} />);

  // Default hero displays latest close ($399.00)
  expect(container.querySelector('.t212-price')).toHaveTextContent('$399.00');
  expect(typeof lastChartProps.options.onHover).toBe('function');

  // 1. Hover historical bar index 5 (1M slice starts at close=378, index 5 => close=383)
  act(() => {
    lastChartProps.options.onHover({ type: 'mousemove' }, [{ datasetIndex: 0, index: 5 }]);
  });
  expect(container.querySelector('.t212-price')).toHaveTextContent('$383.00');
  expect(container.querySelector('.t212-scrub-date')).not.toBeNull();

  // 2. Hover forecast bar index 24 (splitIndex=21, offset=2 => futureDates[2]='2024-06-05', midpoint=403, range=399..407)
  act(() => {
    lastChartProps.options.onHover({ type: 'mousemove' }, [{ datasetIndex: 1, index: 24 }]);
  });
  expect(container.querySelector('.t212-price')).toHaveTextContent('$403.00');
  expect(container.querySelector('.t212-scrub-date')).toHaveTextContent('2024-06-05');
  expect(container.querySelector('.t212-scrub-range')).toHaveTextContent('$399.00 – $407.00');

  // 3. Pointer leave restores latest close cleanly
  fireEvent.pointerLeave(container.querySelector('.t212-chart-wrap'));
  expect(container.querySelector('.t212-price')).toHaveTextContent('$399.00');
  expect(container.querySelector('.t212-scrub-date')).toBeNull();
});
