import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { usePriceHistory } from '../hooks/usePriceHistory';
import {
  CHART_RANGES,
  availableRanges,
  defaultRangeId,
  periodChange,
  sliceRangePoints,
} from '../utils/priceRanges';
import { getSharedEstimatePresentation } from '../utils/estimateAdapter';
import LazyLineChart from './LazyLineChart';

const MIN_ZOOM_POINTS = 12;
const COMPANY_NAME_FALLBACKS = { AIG: 'American International Group' };

function themeIsDark() {
  if (typeof document === 'undefined') return true;
  return (document.documentElement.getAttribute('data-theme') || 'dark') !== 'light';
}

function useAppTheme() {
  const [isDark, setIsDark] = useState(themeIsDark);
  useEffect(() => {
    if (typeof document === 'undefined' || typeof MutationObserver === 'undefined') return undefined;
    const observer = new MutationObserver(() => setIsDark(themeIsDark()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);
  return isDark;
}

function useIsMobileViewport() {
  const [isMobile, setIsMobile] = useState(() => (
    typeof window !== 'undefined' && typeof window.matchMedia === 'function'
      ? window.matchMedia('(max-width: 640px)').matches
      : false
  ));
  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return undefined;
    const media = window.matchMedia('(max-width: 640px)');
    const onChange = (event) => setIsMobile(event.matches);
    setIsMobile(media.matches);
    if (typeof media.addEventListener === 'function') {
      media.addEventListener('change', onChange);
      return () => media.removeEventListener('change', onChange);
    }
    media.addListener?.(onChange);
    return () => media.removeListener?.(onChange);
  }, []);
  return isMobile;
}

function formatMoneyLocal(value, currencySymbol) {
  if (value == null || value === '' || !Number.isFinite(Number(value))) return '—';
  const isPence = currencySymbol === 'p' || currencySymbol === 'GBp';
  const symbol = isPence ? 'p' : (currencySymbol || '$');
  const decimals = isPence ? 1 : 2;
  const numeric = Number(value);
  const magnitude = Math.abs(numeric).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  if (isPence) {
    return `${numeric < 0 ? '-' : ''}${magnitude}p`;
  }
  return `${numeric < 0 ? '-' : ''}${symbol}${magnitude}`;
}

export function formatAxisLabel(label, isIntraday, isMultiYear = false) {
  if (label == null) return '';
  if (!isIntraday) {
    if (typeof label === 'string' && label.length >= 10 && label.includes('-')) {
      return isMultiYear ? label.slice(0, 7) : label.slice(5);
    }
    return String(label);
  }
  const parsed = Date.parse(label);
  if (!Number.isFinite(parsed)) return String(label);
  const date = new Date(parsed);
  return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
}

function padAnchoredForecast(values, anchorValue, splitIndex) {
  const leadNulls = Math.max(0, splitIndex);
  return [...Array(leadNulls).fill(null), anchorValue, ...(values || [])];
}

// Vertical crosshair through the hovered point
const crosshairPlugin = {
  id: 't212Crosshair',
  afterDraw: (chart) => {
    const active = chart.tooltip?.getActiveElements?.();
    if (!active || active.length === 0) return;
    const { ctx, chartArea } = chart;
    const x = active[0].element.x;
    if (x < chartArea.left || x > chartArea.right) return;
    ctx.save();
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = chart.config.options?.isDark ? 'rgba(255,255,255,0.25)' : 'rgba(0,0,0,0.25)';
    ctx.lineWidth = 1;
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();
    ctx.restore();
  },
};

// Dashed last-price line with a value tag on the right edge
const lastPricePlugin = {
  id: 't212LastPrice',
  afterDraw: (chart) => {
    const price = chart.config.options?.plugins?.t212LastPrice?.value;
    if (!Number.isFinite(price) || !chart.scales?.y || !chart.chartArea) return;
    const { ctx, chartArea, scales } = chart;
    const y = scales.y.getPixelForValue(price);
    if (y < chartArea.top || y > chartArea.bottom) return;
    const color = chart.config.options?.plugins?.t212LastPrice?.color || '#8b93a7';
    const label = chart.config.options?.plugins?.t212LastPrice?.label || '';
    ctx.save();
    ctx.beginPath();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.moveTo(chartArea.left, y);
    ctx.lineTo(chartArea.right, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = '600 10px Inter, sans-serif';
    const width = ctx.measureText(label).width + 12;
    const tagY = Math.min(Math.max(y - 9, chartArea.top), chartArea.bottom - 18);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(chartArea.right - width, tagY, width, 18, 4);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, chartArea.right - width / 2, tagY + 9);
    ctx.restore();
  },
};

// Shaded estimate region past the last actual
const forecastRegionPlugin = {
  id: 't212ForecastRegion',
  beforeDraw: (chart) => {
    const { ctx, chartArea, scales } = chart;
    const splitIdx = chart.options.plugins?.t212ForecastRegion?.splitIndex;
    if (splitIdx == null || !scales.x || !chartArea) return;
    const xPos = scales.x.getPixelForValue(splitIdx);
    if (xPos == null || xPos < chartArea.left || xPos > chartArea.right) return;
    ctx.save();
    ctx.fillStyle = chart.config.options?.isDark ? 'rgba(56,189,248,0.06)' : 'rgba(2,132,199,0.06)';
    ctx.fillRect(xPos, chartArea.top, chartArea.right - xPos, chartArea.bottom - chartArea.top);
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = chart.config.options?.isDark ? 'rgba(56,189,248,0.35)' : 'rgba(2,132,199,0.35)';
    ctx.lineWidth = 1.5;
    ctx.moveTo(xPos, chartArea.top);
    ctx.lineTo(xPos, chartArea.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = chart.config.options?.isDark ? '#38bdf8' : '#0284c7';
    ctx.font = '600 10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('ESTIMATE', xPos + 6, chartArea.top + 14);
    ctx.restore();
  },
};

const CHART_PLUGINS = [crosshairPlugin, lastPricePlugin, forecastRegionPlugin];

export default function PriceChart({
  ticker,
  currencySymbol = '$',
  forecast = null,
  companyName = null,
  onHistorySettled = null,
  onRetryForecast = null,
  refreshToken = 0,
  marketHistory,
  historyLoading,
  historyError,
  onRetryHistory = null,
}) {
  const isDark = useAppTheme();
  const isMobile = useIsMobileViewport();
  const isControlledHistory = marketHistory !== undefined
    || historyLoading !== undefined
    || historyError !== undefined
    || onRetryHistory !== null;
  const hookHistory = usePriceHistory(isControlledHistory ? null : ticker, refreshToken);
  const history = isControlledHistory ? (marketHistory ?? null) : hookHistory.history;
  const loading = isControlledHistory ? Boolean(historyLoading) : hookHistory.loading;
  const error = isControlledHistory ? (historyError || '') : hookHistory.error;
  const meta = isControlledHistory ? (marketHistory?.meta ?? null) : hookHistory.meta;
  const retry = isControlledHistory ? (onRetryHistory || (() => {})) : hookHistory.retry;

  const descriptionId = React.useId();
  const tablePanelId = `${descriptionId}-table`;
  const [tableOpen, setTableOpen] = useState(false);
  const [rangeId, setRangeId] = useState(null);
  const [view, setView] = useState(null);
  const [isInteracting, setIsInteracting] = useState(false);
  const [showForecast, setShowForecast] = useState(true);
  const [isExpanded, setIsExpanded] = useState(false);
  const [showEstimateInfo, setShowEstimateInfo] = useState(false);

  const chartRef = useRef(null);
  const wrapRef = useRef(null);
  const dragRef = useRef(null);
  const reportedRef = useRef(null);
  const expandBtnRef = useRef(null);

  const forecastForTicker = forecast && forecast.ticker === String(ticker || '').toUpperCase() ? forecast : null;

  // Degraded mode: fallback history embedded in forecast payload if primary endpoint fails
  const fallbackHistory = useMemo(() => {
    if (forecastForTicker?.historical_provenance === 'synthetic') return null;
    const dates = forecastForTicker?.historical_dates;
    const prices = forecastForTicker?.historical_prices;
    if (!Array.isArray(dates) || !Array.isArray(prices)) return null;
    const count = Math.min(dates.length, prices.length);
    const daily = [];
    for (let i = 0; i < count; i += 1) {
      if (prices[i] == null || prices[i] === '') continue;
      const close = Number(prices[i]);
      if (dates[i] && Number.isFinite(close) && close > 0) daily.push({ d: String(dates[i]), c: close });
    }
    if (daily.length < 2) return null;
    return {
      ticker: forecastForTicker.ticker,
      asOf: forecastForTicker.data_as_of,
      daily,
      intraday: null,
      degraded: true,
    };
  }, [forecastForTicker]);

  const effectiveHistory = history || fallbackHistory;
  const showLoading = loading && !fallbackHistory;
  const showError = !loading && error && !fallbackHistory;

  // Single presentation adapter verification
  const estimate = useMemo(() => {
    if (!forecastForTicker) return null;
    return getSharedEstimatePresentation({
      forecast: forecastForTicker,
      history: effectiveHistory,
      currencySymbol,
    });
  }, [forecastForTicker, effectiveHistory, currencySymbol]);

  useEffect(() => {
    if (!onHistorySettled) return;
    const key = `${String(ticker || '').toUpperCase()}:${refreshToken}`;
    if (effectiveHistory && reportedRef.current !== `${key}:ok`) {
      reportedRef.current = `${key}:ok`;
      onHistorySettled({
        ok: true,
        degraded: !history && Boolean(fallbackHistory),
        fetchMs: meta?.fetchMs ?? null,
        fromCache: Boolean(meta?.fromCache),
        marketDataCache: history?.marketDataCache ?? null,
      });
    } else if (showError && reportedRef.current !== `${key}:ok` && reportedRef.current !== `${key}:error`) {
      reportedRef.current = `${key}:error`;
      onHistorySettled({ ok: false, degraded: false, fetchMs: null, fromCache: false, marketDataCache: null });
    }
  }, [effectiveHistory, showError, history, fallbackHistory, meta, ticker, refreshToken, onHistorySettled]);

  const hasIntraday = Array.isArray(effectiveHistory?.intraday) && effectiveHistory.intraday.length >= 5;
  const dailyCount = Array.isArray(effectiveHistory?.daily) ? effectiveHistory.daily.length : 0;
  const available = useMemo(
    () => (effectiveHistory ? availableRanges({ dailyCount, hasIntraday }) : []),
    [effectiveHistory, dailyCount, hasIntraday],
  );

  useEffect(() => {
    setRangeId((current) => {
      const avail = availableRanges({ dailyCount, hasIntraday });
      const next = effectiveHistory ? defaultRangeId(avail) : null;
      return current && avail.includes(current) ? current : next;
    });
    setView(null);
    setIsInteracting(false);
  }, [effectiveHistory, dailyCount, hasIntraday, ticker]);

  const points = useMemo(
    () => (effectiveHistory && rangeId
      ? sliceRangePoints(effectiveHistory, rangeId)
      : { labels: [], prices: [], isIntraday: false }),
    [effectiveHistory, rangeId],
  );

  const activeRange = rangeId || defaultRangeId(available);
  const isIntradayRange = activeRange === '24H' || points.isIntraday;
  const isMultiYearRange = activeRange === '5Y' || (
    activeRange === 'MAX'
    && points.labels.length > 1
    && String(points.labels[0]).slice(0, 4) !== String(points.labels.at(-1)).slice(0, 4)
  );

  // Overlay forecast only when enabled, not intraday, and estimate is available without mismatch
  const canShowForecastOverlay = showForecast && !isIntradayRange && estimate?.isAvailable;

  const totalLabels = points.labels.length + (canShowForecastOverlay ? estimate.futureDates.length : 0);
  const fullView = useMemo(
    () => (totalLabels > 0 ? { start: 0, end: totalLabels - 1 } : null),
    [totalLabels],
  );
  const effectiveView = view || fullView;
  const isZoomed = Boolean(view && fullView && (view.start > 0 || view.end < fullView.end));
  const canZoomIn = Boolean(
    effectiveView
    && fullView
    && (effectiveView.end - effectiveView.start + 1) > Math.min(MIN_ZOOM_POINTS, totalLabels),
  );

  const stats = useMemo(() => {
    const change = periodChange(points.prices);
    const firstFinite = points.prices.find(Number.isFinite) ?? null;
    const lastFinite = points.prices.filter(Number.isFinite).at(-1);
    return { ...change, first: firstFinite, last: lastFinite };
  }, [points]);

  const hasData = points.labels.length > 0 && Number.isFinite(stats.last);

  const colors = useMemo(() => {
    const up = stats.up;
    if (isDark) {
      return {
        line: up ? '#00f5a0' : '#ff5c5c',
        areaTop: up ? 'rgba(0,245,160,0.18)' : 'rgba(255,92,92,0.18)',
        grid: 'rgba(255,255,255,0.05)',
        tick: '#94a3b8',
        tooltipBg: '#0d0d1a',
        tooltipTitle: '#e8e8f0',
        tooltipBody: '#cbd5e1',
        tooltipBorder: 'rgba(255,255,255,0.12)',
        estimate: '#38bdf8',
        bandBorder: 'rgba(56, 189, 248, 0.28)',
        bandFill: 'rgba(56, 189, 248, 0.12)',
      };
    }
    return {
      line: up ? '#10b981' : '#ef4444',
      areaTop: up ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)',
      grid: 'rgba(0,0,0,0.06)',
      tick: '#64748b',
      tooltipBg: '#ffffff',
      tooltipTitle: '#1e293b',
      tooltipBody: '#475569',
      tooltipBorder: 'rgba(0,0,0,0.08)',
      estimate: '#0284c7',
      bandBorder: 'rgba(2, 132, 199, 0.28)',
      bandFill: 'rgba(2, 132, 199, 0.12)',
    };
  }, [isDark, stats.up]);

  const chartData = useMemo(() => {
    const labels = [...points.labels];
    const historyPadded = [...points.prices];
    const datasets = [];

    // Main historical price dataset
    datasets.push({
      role: 'history',
      label: 'Price',
      data: historyPadded,
      borderColor: colors.line,
      backgroundColor: (context) => {
        const { ctx, chartArea } = context.chart;
        if (!chartArea) return colors.areaTop;
        const grad = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
        grad.addColorStop(0, colors.areaTop);
        grad.addColorStop(1, 'transparent');
        return grad;
      },
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 5,
      pointHoverBackgroundColor: colors.line,
      tension: 0.25,
      fill: true,
      spanGaps: false,
    });

    let forecastSplitIndex = null;

    if (canShowForecastOverlay && estimate?.series?.length > 0 && points.prices.length > 0) {
      estimate.futureDates.forEach((futureDate) => {
        historyPadded.push(null);
        labels.push(futureDate);
      });

      forecastSplitIndex = points.prices.length - 1;
      const anchorClose = points.prices[forecastSplitIndex];
      const hasRangeBand = Array.isArray(estimate.lowerSeries)
        && Array.isArray(estimate.upperSeries)
        && estimate.lowerSeries.length === estimate.series.length
        && estimate.upperSeries.length === estimate.series.length;

      if (hasRangeBand) {
        datasets.push({
          role: 'band-upper',
          label: 'Historical error range (experimental)',
          data: padAnchoredForecast(estimate.upperSeries, anchorClose, forecastSplitIndex),
          borderColor: colors.bandBorder,
          backgroundColor: colors.bandFill,
          borderWidth: 1,
          pointRadius: 0,
          pointHoverRadius: 0,
          pointHitRadius: 0,
          tension: 0.25,
          fill: '+1',
          spanGaps: false,
        });

        datasets.push({
          role: 'band-lower',
          label: 'Historical error range (experimental) — lower',
          data: padAnchoredForecast(estimate.lowerSeries, anchorClose, forecastSplitIndex),
          borderColor: colors.bandBorder,
          backgroundColor: 'transparent',
          borderWidth: 1,
          pointRadius: 0,
          pointHoverRadius: 0,
          pointHitRadius: 0,
          tension: 0.25,
          fill: false,
          spanGaps: false,
        });
      }

      datasets.push({
        role: 'midpoint',
        label: 'Range midpoint — experimental',
        data: padAnchoredForecast(estimate.series, anchorClose, forecastSplitIndex),
        borderColor: colors.estimate,
        backgroundColor: 'transparent',
        borderWidth: 2.2,
        borderDash: [6, 4],
        pointRadius: 0,
        pointHoverRadius: 6,
        pointHoverBackgroundColor: colors.estimate,
        tension: 0.25,
        fill: false,
        spanGaps: false,
      });
    }

    return {
      labels,
      forecastSplitIndex,
      datasets,
    };
  }, [points, canShowForecastOverlay, estimate, colors]);

  const datasetByRole = useMemo(() => ({
    history: chartData.datasets.find((ds) => ds.role === 'history') || null,
    midpoint: chartData.datasets.find((ds) => ds.role === 'midpoint') || null,
    bandUpper: chartData.datasets.find((ds) => ds.role === 'band-upper') || null,
    bandLower: chartData.datasets.find((ds) => ds.role === 'band-lower') || null,
  }), [chartData]);

  const yBounds = useMemo(() => {
    if (!effectiveView || chartData.labels.length === 0) return {};
    const values = [];
    for (const ds of chartData.datasets) {
      (ds.data || []).slice(effectiveView.start, effectiveView.end + 1).forEach((value) => {
        if (Number.isFinite(value)) values.push(value);
      });
    }
    if (values.length === 0) return {};
    const min = Math.min(...values);
    const max = Math.max(...values);
    const padding = Math.max((max - min) * 0.12, max * 0.015);
    return { suggestedMin: Math.max(0, min - padding), suggestedMax: max + padding };
  }, [chartData, effectiveView]);

  const clampView = useCallback((start, end) => {
    if (!fullView) return null;
    const size = fullView.end - fullView.start + 1;
    let next = { start: Math.round(start), end: Math.round(end) };
    if (next.end - next.start + 1 < Math.min(MIN_ZOOM_POINTS, size)) {
      const mid = (next.start + next.end) / 2;
      const half = Math.floor(Math.min(MIN_ZOOM_POINTS, size) / 2);
      next = { start: Math.round(mid) - half, end: Math.round(mid) + half };
    }
    if (next.start < fullView.start) {
      next = { start: fullView.start, end: Math.min(fullView.end, fullView.start + (next.end - next.start)) };
    }
    if (next.end > fullView.end) {
      next = { end: fullView.end, start: Math.max(fullView.start, fullView.end - (next.end - next.start)) };
    }
    if (next.start <= fullView.start && next.end >= fullView.end) return null;
    return next;
  }, [fullView]);

  const effectiveViewRef = useRef(effectiveView);
  effectiveViewRef.current = effectiveView;
  const fullViewRef = useRef(fullView);
  fullViewRef.current = fullView;
  const clampViewRef = useRef(clampView);
  clampViewRef.current = clampView;
  const isExpandedRef = useRef(isExpanded);
  isExpandedRef.current = isExpanded;

  const zoomAt = useCallback((pixelX, factor) => {
    const chart = chartRef.current;
    const currentView = effectiveViewRef.current;
    const currentFull = fullViewRef.current;
    if (!chart || !currentView || !currentFull || factor <= 0) return;
    const area = chart.chartArea;
    const clampedX = Math.min(Math.max(pixelX, area.left), area.right);
    const cursor = chart.scales.x.getValueForPixel(clampedX);
    const size = currentView.end - currentView.start + 1;
    const nextSize = Math.max(
      Math.min(MIN_ZOOM_POINTS, currentFull.end - currentFull.start + 1),
      Math.round(size * factor),
    );
    const ratio = size <= 1 ? 0.5 : (cursor - currentView.start) / (size - 1);
    const nextStart = cursor - ratio * (nextSize - 1);
    const nextView = clampViewRef.current(nextStart, nextStart + nextSize - 1);
    setIsInteracting(Boolean(nextView));
    setView(nextView);
  }, []);

  const zoomByCenter = useCallback((factor) => {
    const currentView = effectiveViewRef.current;
    const currentFull = fullViewRef.current;
    if (!currentView || !currentFull || factor <= 0) return;
    const size = currentView.end - currentView.start + 1;
    const fullSize = currentFull.end - currentFull.start + 1;
    const nextSize = Math.max(Math.min(MIN_ZOOM_POINTS, fullSize), Math.round(size * factor));
    const center = (currentView.start + currentView.end) / 2;
    const nextStart = center - (nextSize - 1) / 2;
    setIsInteracting(true);
    const nextView = clampViewRef.current(nextStart, nextStart + nextSize - 1);
    setView(nextView);
    if (!nextView) setIsInteracting(false);
  }, []);

  useEffect(() => {
    const node = wrapRef.current;
    if (!node) return undefined;
    const onWheel = (event) => {
      if (!chartRef.current) return;
      if (!event.ctrlKey && !event.metaKey && !isExpandedRef.current) return;
      event.preventDefault();
      const rect = node.getBoundingClientRect();
      zoomAt(event.clientX - rect.left, event.deltaY > 0 ? 1.18 : 0.85);
    };
    const onPointerDown = (event) => {
      if (event.pointerType === 'touch') return;
      if (event.button !== 0 || !chartRef.current) return;
      if (event.target instanceof Element && event.target.closest('button')) return;
      const rect = node.getBoundingClientRect();
      dragRef.current = { x: event.clientX - rect.left, view: effectiveViewRef.current };
      setIsInteracting(true);
      node.setPointerCapture?.(event.pointerId);
    };
    const onPointerMove = (event) => {
      const drag = dragRef.current;
      const chart = chartRef.current;
      if (!drag || !chart || !drag.view) return;
      const rect = node.getBoundingClientRect();
      const dx = event.clientX - rect.left - drag.x;
      const areaWidth = chart.chartArea.width || 1;
      const size = drag.view.end - drag.view.start + 1;
      const shift = Math.round((dx / areaWidth) * size);
      if (shift !== 0) setView(clampViewRef.current(drag.view.start - shift, drag.view.end - shift));
    };
    const endDrag = () => {
      if (dragRef.current) {
        dragRef.current = null;
        setIsInteracting(false);
      }
    };
    const onDoubleClick = () => {
      setView(null);
      setIsInteracting(false);
    };
    node.addEventListener('wheel', onWheel, { passive: false });
    node.addEventListener('pointerdown', onPointerDown);
    node.addEventListener('pointermove', onPointerMove);
    node.addEventListener('pointerup', endDrag);
    node.addEventListener('pointercancel', endDrag);
    node.addEventListener('lostpointercapture', endDrag);
    node.addEventListener('dblclick', onDoubleClick);
    return () => {
      node.removeEventListener('wheel', onWheel);
      node.removeEventListener('pointerdown', onPointerDown);
      node.removeEventListener('pointermove', onPointerMove);
      node.removeEventListener('pointerup', endDrag);
      node.removeEventListener('pointercancel', endDrag);
      node.removeEventListener('lostpointercapture', endDrag);
      node.removeEventListener('dblclick', onDoubleClick);
    };
  }, [zoomAt]);

  // Modal Escape listener & focus restoration
  useEffect(() => {
    if (!isExpanded) return undefined;
    const dialog = wrapRef.current;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    // Make every sibling outside the dialog inert, preserving prior values.
    const blocked = [];
    for (let node = dialog; node?.parentElement; node = node.parentElement) {
      for (const sibling of node.parentElement.children) {
        if (sibling !== node) {
          blocked.push([sibling, sibling.inert]);
          sibling.inert = true;
        }
      }
      if (node.parentElement === document.body) break;
    }
    const focusable = () => [...dialog.querySelectorAll('button:not(:disabled), [href], [tabindex="0"]')];
    focusable()[0]?.focus();
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        setIsExpanded(false);
      }
      if (e.key === 'Tab') {
        const elements = focusable();
        if (!elements.length) return;
        const first = elements[0];
        const last = elements.at(-1);
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault(); last?.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault(); first?.focus();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      blocked.forEach(([element, inert]) => { element.inert = inert; });
      document.body.style.overflow = previousOverflow;
      expandBtnRef.current?.focus();
    };
  }, [isExpanded]);

  const toggleExpand = useCallback(() => {
    setIsExpanded((prev) => {
      if (prev) {
        setTimeout(() => expandBtnRef.current?.focus(), 0);
      }
      return !prev;
    });
  }, []);

  const handleChartErrorChange = useCallback((hasError) => {
    if (hasError) setTableOpen(true);
  }, []);

  const chartOptions = useMemo(() => {
    const firstFinitePrice = stats.first;
    const latestClosePrice = stats.last;
    const bandLowerData = datasetByRole.bandLower?.data || null;
    const bandUpperData = datasetByRole.bandUpper?.data || null;
    const splitIndex = chartData.forecastSplitIndex;
    const labels = chartData.labels;

    return {
      responsive: true,
      maintainAspectRatio: false,
      isDark,
      animation: (isInteracting || isZoomed) ? false : { duration: 250 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        t212ForecastRegion: { splitIndex },
        t212LastPrice: canShowForecastOverlay
          ? undefined
          : {
            value: latestClosePrice,
            color: stats.change > 0 ? (isDark ? '#00f5a0' : '#10b981') : stats.change < 0 ? (isDark ? '#ff5c5c' : '#ef4444') : '#8b93a7',
            label: formatMoneyLocal(latestClosePrice, currencySymbol),
          },
        tooltip: {
          backgroundColor: colors.tooltipBg,
          titleColor: colors.tooltipTitle,
          bodyColor: colors.tooltipBody,
          borderColor: colors.tooltipBorder,
          borderWidth: 1,
          cornerRadius: 10,
          padding: 12,
          displayColors: false,
          titleFont: { family: 'Inter, sans-serif', size: 12, weight: '600' },
          bodyFont: { family: 'JetBrains Mono, monospace', size: 12 },
          filter: (tooltipItem) => {
            const role = tooltipItem.dataset?.role;
            if (role === 'band-upper' || role === 'band-lower') return false;
            if (role === 'midpoint' && splitIndex != null && tooltipItem.dataIndex <= splitIndex) return false;
            return true;
          },
          callbacks: {
            title: (items) => (items.length > 0 ? String(labels[items[0].dataIndex] ?? '') : ''),
            label: (ctx) => {
              const val = Number(ctx.parsed.y);
              if (!Number.isFinite(val)) return null;
              const isMidpoint = ctx.dataset?.role === 'midpoint';
              const basePrice = isMidpoint ? latestClosePrice : firstFinitePrice;
              const rel = Number.isFinite(basePrice) && basePrice > 0 ? ((val / basePrice - 1) * 100) : 0;
              return ` ${ctx.dataset.label}: ${formatMoneyLocal(val, currencySymbol)} (${rel >= 0 ? '+' : ''}${rel.toFixed(2)}%)`;
            },
            afterLabel: (ctx) => {
              if (ctx.dataset?.role !== 'midpoint') return '';
              const low = bandLowerData?.[ctx.dataIndex];
              const high = bandUpperData?.[ctx.dataIndex];
              if (!Number.isFinite(low) || !Number.isFinite(high)) return '';
              return ` Range: ${formatMoneyLocal(low, currencySymbol)} – ${formatMoneyLocal(high, currencySymbol)}`;
            },
          },
        },
      },
      scales: {
        x: {
          min: effectiveView?.start,
          max: effectiveView?.end,
          ticks: {
            color: colors.tick,
            maxTicksLimit: isMobile ? 4 : 7,
            maxRotation: 0,
            autoSkip: true,
            autoSkipPadding: 12,
            font: { family: 'JetBrains Mono, monospace', size: 10 },
            callback(value) {
              const rawLabel = typeof this?.getLabelForValue === 'function'
                ? this.getLabelForValue(value)
                : labels[value];
              return formatAxisLabel(rawLabel, isIntradayRange, isMultiYearRange);
            },
          },
          grid: { display: false },
        },
        y: {
          ...yBounds,
          position: 'right',
          ticks: {
            color: colors.tick,
            font: { family: 'JetBrains Mono, monospace', size: 10 },
            callback: (v) => formatMoneyLocal(Number(v), currencySymbol),
          },
          grid: { color: colors.grid },
        },
      },
    };
  }, [
    isDark,
    isMobile,
    isInteracting,
    isZoomed,
    colors,
    chartData,
    datasetByRole,
    effectiveView,
    yBounds,
    stats.first,
    stats.last,
    stats.change,
    currencySymbol,
    canShowForecastOverlay,
    isIntradayRange,
    isMultiYearRange,
  ]);

  const selectRange = useCallback((id) => {
    setRangeId(id);
    setView(null);
    setIsInteracting(false);
  }, []);

  const resetZoom = useCallback(() => {
    setView(null);
    setIsInteracting(false);
  }, []);

  const changeClass = stats.change > 0 ? 'up' : stats.change < 0 ? 'down' : 'flat';
  const sign = stats.change > 0 ? '+' : '';
  const dataDate = effectiveHistory?.asOf || (Array.isArray(effectiveHistory?.daily) ? effectiveHistory.daily.at(-1)?.d : null);
  const suppliedCompanyName = (companyName || forecastForTicker?.ticker_name || '').trim();
  const resolvedCompanyName = suppliedCompanyName && suppliedCompanyName.toUpperCase() !== String(ticker || '').toUpperCase()
    ? suppliedCompanyName
    : COMPANY_NAME_FALLBACKS[String(ticker || '').toUpperCase()] || null;
  const finalLow = estimate?.lowerSeries?.at(-1);
  const finalHigh = estimate?.upperSeries?.at(-1);
  const hasEstimateRange = Number.isFinite(finalLow) && Number.isFinite(finalHigh);

  const renderRangeAndOverlayControls = (inModal = false) => (
    <>
      <div className="t212-range-tabs" role="group" aria-label={inModal ? 'Expanded chart time range' : 'Chart time range'}>
        {CHART_RANGES.filter((range) => available.includes(range.id)).map((range) => (
          <button
            key={range.id}
            type="button"
            aria-pressed={activeRange === range.id}
            className={`range-pill ${activeRange === range.id ? 'active' : ''}`}
            onClick={() => selectRange(range.id)}
          >
            {range.label}
          </button>
        ))}
      </div>

      <div className="t212-toolbar-actions">
        <div className="estimate-toggle-wrapper">
          <button
            type="button"
            className={`toolbar-btn estimate-toggle-btn ${showForecast && !isIntradayRange ? 'active' : ''}`}
            disabled={isIntradayRange}
            onClick={() => setShowForecast((prev) => !prev)}
            title={isIntradayRange ? '7-session midpoint is hidden on 24H intraday range' : 'Toggle 7-session midpoint and historical error range'}
            aria-pressed={showForecast && !isIntradayRange}
          >
            <span className="estimate-indicator-dash" aria-hidden="true" />
            7-session midpoint
          </button>
          {isIntradayRange && (
            <span className="intraday-estimate-note">Hidden for 24H</span>
          )}
        </div>

        <div className="zoom-btn-group" role="group" aria-label="Chart zoom">
          <button
            type="button"
            className="toolbar-btn zoom-btn"
            onClick={() => zoomByCenter(0.75)}
            disabled={!canZoomIn}
            aria-label="Zoom in"
            title="Zoom in (or Ctrl/Cmd + scroll)"
          >
            +
          </button>
          <button
            type="button"
            className="toolbar-btn zoom-btn"
            onClick={() => zoomByCenter(1.35)}
            disabled={!isZoomed}
            aria-label="Zoom out"
            title="Zoom out"
          >
            −
          </button>
        </div>

        {isZoomed && (
          <button
            type="button"
            className="toolbar-btn reset-btn"
            onClick={resetZoom}
            aria-label="Reset"
          >
            Reset
          </button>
        )}

        {!inModal && hasData && (
          <button
            type="button"
            className={`toolbar-btn table-toggle-btn ${tableOpen ? 'active' : ''}`}
            onClick={() => setTableOpen((prev) => !prev)}
            aria-expanded={tableOpen}
            aria-controls={tablePanelId}
          >
            Table
          </button>
        )}

        {!inModal && (
          <button
            ref={expandBtnRef}
            type="button"
            className="toolbar-btn expand-btn"
            onClick={toggleExpand}
            aria-label={isExpanded ? 'Close expanded chart' : 'Expand chart'}
            title={isExpanded ? 'Close (Esc)' : 'Expand chart'}
          >
            {isExpanded ? 'Close' : 'Expand'}
          </button>
        )}
      </div>
    </>
  );

  return (
    <section
      id="chartContainer"
      className={`t212-chart-section ${isExpanded ? 'expanded-modal-open' : ''}`}
      aria-label={`${ticker} price chart`}
    >
      {/* Chart Header */}
      <div className="t212-chart-head">
        <div className="t212-price-block">
          <div className="t212-title-row">
            {resolvedCompanyName && <span className="t212-company-name">{resolvedCompanyName}</span>}
            <span className="t212-ticker">{ticker}</span>
          </div>

          {hasData ? (
            <div className="t212-metrics-row">
              <strong className="t212-price mono">
                {formatMoneyLocal(stats.last, currencySymbol)}
              </strong>
              {points.prices.filter(Number.isFinite).length >= 2 && (
                <span className={`t212-change ${changeClass}`}>
                  {sign}{formatMoneyLocal(stats.change, currencySymbol)} ({sign}{stats.changePct.toFixed(2)}%)
                </span>
              )}
              <span className="t212-range-name">
                {activeRange === 'MAX' ? `Available history · ${points.labels[0]}–${points.labels.at(-1)}` : activeRange}
              </span>
              {dataDate && !points.isIntraday && (
                <span className="t212-data-date" title={`Market data through ${dataDate}`}>
                  Data through {dataDate}
                </span>
              )}
            </div>
          ) : (
            <span className="t212-price-skeleton" aria-hidden="true" />
          )}
        </div>

        {/* Restrained Toolbar */}
        <div className="t212-restrained-toolbar">
          {renderRangeAndOverlayControls(false)}
        </div>
      </div>

      {/* In-Chart Forecast Summary Line */}
      {estimate?.isAvailable && (
        <div className="chart-estimate-bar">
          <div className="estimate-summary-text">
            <span className="estimate-dot" aria-hidden="true" />
            <span>
              <strong>Range midpoint — experimental:</strong> {formatMoneyLocal(estimate.finalPrice, currencySymbol)} ·{' '}
              <span className={`estimate-delta ${estimate.direction}`}>
                {estimate.changePct != null && Number.isFinite(estimate.changePct)
                  ? `${estimate.changePct > 0 ? '+' : ''}${estimate.changePct.toFixed(1)}%`
                  : '—'}
              </span>{' '}
              from the latest close · After 7 trading sessions ({estimate.futureDates.at(-1)})
              {hasEstimateRange && (
                <>
                  {' · '}
                  <span className="estimate-range-summary">
                    Historical error range (experimental): {formatMoneyLocal(finalLow, currencySymbol)} – {formatMoneyLocal(finalHigh, currencySymbol)}
                  </span>
                </>
              )}
            </span>
            <button
              type="button"
              className="info-icon-btn"
              onClick={() => setShowEstimateInfo((prev) => !prev)}
              aria-label="How this estimate works"
              aria-expanded={showEstimateInfo}
              title="How this estimate works"
            >
              ℹ
            </button>
          </div>

          {showEstimateInfo && (
            <p className="estimate-inline-explanation">
              The shaded band shows the historical error range (experimental) anchored at the latest close, and the dashed line averages its upper and lower bounds.
              {estimate.pointEstimate != null ? ` The model point forecast is ${formatMoneyLocal(estimate.pointEstimate, currencySymbol)}.` : ' The model point forecast is unavailable.'}
              {' '}Coverage for this refit is unverified and has not been certified.
            </p>
          )}
        </div>
      )}

      {/* Chart Canvas Wrap */}
      <div
        className={`t212-chart-wrap ${isExpanded ? 'expanded' : ''}`}
        ref={wrapRef}
        role={isExpanded ? 'dialog' : undefined}
        aria-modal={isExpanded ? 'true' : undefined}
        aria-label={isExpanded ? 'Expanded price chart' : undefined}
      >
        {isExpanded && (
          <div className="expanded-modal-header">
            <div className="expanded-modal-bar">
              <span className="expanded-title" id="expanded-chart-title">{ticker} · Interactive chart</span>
              <button
                type="button"
                className="expanded-close-btn"
                onClick={toggleExpand}
                aria-label="Close expanded chart"
              >
                ✕ Close (Esc)
              </button>
            </div>
            <div className="t212-restrained-toolbar expanded-modal-toolbar">
              {renderRangeAndOverlayControls(true)}
            </div>
          </div>
        )}

        {showLoading && (
          <div className="t212-chart-skeleton" role="status" aria-live="polite">
            <span className="t212-skeleton-bars" aria-hidden="true">
              {Array.from({ length: 24 }, (_, index) => (
                <i key={index} style={{ height: `${28 + ((index * 37) % 62)}%` }} />
              ))}
            </span>
            <span className="t212-skeleton-label">Loading price history…</span>
          </div>
        )}

        {showError && (
          <div className="t212-chart-error" role="alert">
            <p>{error}</p>
            <button type="button" className="retry-button" onClick={retry}>Retry chart</button>
          </div>
        )}

        <div className="t212-canvas-viewport">
          {!showLoading && !showError && effectiveHistory && points.labels.length > 0 && (
            <React.Suspense fallback={<div className="loading-text">Loading Chart…</div>}>
              <LazyLineChart
                ref={chartRef}
                role="img"
                aria-label={`${ticker} historical closes${canShowForecastOverlay ? ' and experimental range midpoint' : ''}`}
                aria-describedby={descriptionId}
                data={chartData}
                options={chartOptions}
                plugins={CHART_PLUGINS}
                onErrorChange={handleChartErrorChange}
              />
            </React.Suspense>
          )}

          {!showLoading && !showError && effectiveHistory && points.labels.length === 0 && (
            <div className="empty-copy">Not enough price history for this view.</div>
          )}
        </div>
      </div>

      {hasData && (
        <p id={descriptionId} className="sr-only">
          Solid line: recorded historical closes. Shaded region and dashed line: historical error range (experimental) and seven-session midpoint. Use the Table button in the chart toolbar to read exact values.
        </p>
      )}

      {hasData && tableOpen && (
        <div id={tablePanelId} className="chart-data-table-panel">
          <div className="chart-table-scroll" tabIndex="0">
            <table>
              <caption>{ticker} prices and experimental range midpoint</caption>
              <thead>
                <tr>
                  <th scope="col">Date</th>
                  <th scope="col">Historical close</th>
                  <th scope="col">Range midpoint</th>
                  {datasetByRole.bandLower && datasetByRole.bandUpper && (
                    <th scope="col">Historical error range (experimental)</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {chartData.labels.map((date, index) => {
                  const isForecastPoint = chartData.forecastSplitIndex != null && index > chartData.forecastSplitIndex;
                  const histVal = datasetByRole.history?.data?.[index];
                  const midVal = isForecastPoint ? datasetByRole.midpoint?.data?.[index] : null;
                  const lowVal = isForecastPoint ? datasetByRole.bandLower?.data?.[index] : null;
                  const highVal = isForecastPoint ? datasetByRole.bandUpper?.data?.[index] : null;
                  return (
                    <tr key={`${date}-${index}`}>
                      <th scope="row">{date}</th>
                      <td className="mono">{formatMoneyLocal(histVal, currencySymbol)}</td>
                      <td className="mono">{formatMoneyLocal(midVal, currencySymbol)}</td>
                      {datasetByRole.bandLower && datasetByRole.bandUpper && (
                        <td className="mono">
                          {Number.isFinite(lowVal) && Number.isFinite(highVal)
                            ? `${formatMoneyLocal(lowVal, currencySymbol)} – ${formatMoneyLocal(highVal, currencySymbol)}`
                            : '—'}
                        </td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {estimate?.isMismatch && (
        <div className="chart-mismatch-alert" role="alert">
          <span className="mismatch-icon" aria-hidden="true">⚠️</span>
          <span className="mismatch-msg">
            Forecast data does not match the latest market history ({estimate.mismatchReason}).
          </span>
          {onRetryForecast && (
            <button type="button" className="retry-action-btn" onClick={onRetryForecast}>
              Retry forecast
            </button>
          )}
        </div>
      )}
      {forecastForTicker && estimate && !estimate.isAvailable && !estimate.isMismatch && (
        <p className="chart-mismatch-alert" role="status">
          Seven-day estimate unavailable: forecast data is incomplete or invalid.
          {onRetryForecast && <button type="button" className="retry-action-btn" onClick={onRetryForecast}>Retry forecast</button>}
        </p>
      )}
    </section>
  );
}
