import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchSimpleForecast, fetchTickerNews, wakeForecastService } from './api/simpleForecastClient';
import { fetchPriceHistory } from './api/priceHistoryClient';
import { fetchStockInfo, formatCompactMarketCap } from './api/stockInfoClient';
import { assertTickerIdentity, assertMarketIdentity } from './api/request';
import StockAutocomplete from './components/StockAutocomplete';
import PriceChart from './components/PriceChart';
import VolatilityOutlook from './components/VolatilityOutlook';
import ForecastLedgerTrackRecord from './components/ForecastLedgerTrackRecord';
import { ALL_VALID_TICKERS, ALL_TICKERS_SET } from './universe';

const VALID_TABS = new Set(['overview', 'news', 'performance']);
const VALID_RANGES = new Set(['24H', '5D', '1M', '6M', '1Y', '5Y', 'MAX']);

export function parseDeepLinkSearch(search = '') {
  const params = new URLSearchParams(search || '');
  const rawTicker = String(params.get('ticker') || '').trim().toUpperCase();
  const ticker = rawTicker && ALL_TICKERS_SET.has(rawTicker) ? rawTicker : null;
  const rawTab = String(params.get('tab') || '').trim().toLowerCase();
  const tab = VALID_TABS.has(rawTab) ? rawTab : 'overview';
  const rawRange = String(params.get('range') || '').trim().toUpperCase();
  const range = VALID_RANGES.has(rawRange) ? rawRange : null;
  return { ticker, tab, range };
}

export function buildDeepLinkUrl({ ticker, range, tab } = {}) {
  const pathname = typeof window !== 'undefined' && window.location?.pathname ? window.location.pathname : '/';
  if (!ticker) return pathname;
  const params = new URLSearchParams();
  params.set('ticker', String(ticker).trim().toUpperCase());
  params.set('range', VALID_RANGES.has(String(range || '').toUpperCase()) ? String(range).toUpperCase() : '1M');
  params.set('tab', VALID_TABS.has(String(tab || '').toLowerCase()) ? String(tab).toLowerCase() : 'overview');
  return `${pathname}?${params.toString()}`;
}

function formatSharePrice(value, currencySymbol = '$') {
  if (value == null || value === '' || !Number.isFinite(Number(value))) return '—';
  const isPence = currencySymbol === 'p' || currencySymbol === 'GBp';
  const symbol = isPence ? 'p' : (currencySymbol || '$');
  const decimals = isPence ? 1 : 2;
  const numeric = Number(value);
  const formatted = Math.abs(numeric).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  if (isPence) return `${numeric < 0 ? '-' : ''}${formatted}p`;
  return `${numeric < 0 ? '-' : ''}${symbol}${formatted}`;
}

function formatPercent(value, digits = 1) {
  if (value == null || value === '' || !Number.isFinite(Number(value))) return '—';
  return `${Number(value).toFixed(digits)}%`;
}

function ServiceBadge({ status, attempt, onRetry }) {
  const copy = {
    checking: `Starting forecast service${attempt > 1 ? ` · attempt ${attempt}` : ''}…`,
    online: 'Forecast service reachable',
    offline: 'Forecast service unavailable',
  }[status];
  return (
    <div className={`service-badge ${status}`} role="status" aria-label={copy}>
      <span aria-hidden="true" />
      {copy}
      {status === 'offline' && onRetry && (
        <button type="button" className="retry-action-btn service-retry-btn" onClick={onRetry}>
          Retry connection
        </button>
      )}
    </div>
  );
}

function CompanyFundamentalsStrip({ info, ticker, currencySymbol }) {
  if (!info || info.ticker !== ticker || !info.hasAnyFundamental) return null;
  const effectiveSymbol = info.currencySymbol || currencySymbol;
  const marketCapText = formatCompactMarketCap(info.marketCap, {
    currency: info.currency,
    marketCapCurrency: info.marketCapCurrency,
    currencySymbol: effectiveSymbol,
    ticker,
  });
  const range52wText = info.fiftyTwoWeekLow != null && info.fiftyTwoWeekHigh != null
    ? `${formatSharePrice(info.fiftyTwoWeekLow, effectiveSymbol)} – ${formatSharePrice(info.fiftyTwoWeekHigh, effectiveSymbol)}`
    : '—';
  const peText = info.peRatio != null && Number.isFinite(Number(info.peRatio))
    ? `${Number(info.peRatio).toFixed(1)}×`
    : '—';

  return (
    <section className="panel fundamentals-strip" aria-label={`${ticker} company fundamentals`}>
      <div className="fundamentals-grid">
        <div className="fundamental-item">
          <span className="fundamental-label">Sector</span>
          <strong className="fundamental-value">{info.sector || '—'}</strong>
        </div>
        <div className="fundamental-item">
          <span className="fundamental-label">Market Cap</span>
          <strong className="fundamental-value mono">{marketCapText}</strong>
        </div>
        <div className="fundamental-item">
          <span className="fundamental-label">52-Week Range</span>
          <strong className="fundamental-value mono">{range52wText}</strong>
        </div>
        <div className="fundamental-item">
          <span className="fundamental-label">P/E (TTM)</span>
          <strong className="fundamental-value mono">{peText}</strong>
        </div>
      </div>
    </section>
  );
}

function BacktestPanel({ backtest }) {
  if (!backtest) return null;
  const hasRatio = backtest.relative_mae_vs_persistence != null && backtest.relative_mae_vs_persistence !== '' && Number.isFinite(Number(backtest.relative_mae_vs_persistence));
  const ratio = hasRatio ? Number(backtest.relative_mae_vs_persistence) : null;
  const beatBaseline = hasRatio && ratio < 1;
  const directionAcc = backtest.direction_accuracy != null && backtest.direction_accuracy !== '' && Number.isFinite(Number(backtest.direction_accuracy)) ? Number(backtest.direction_accuracy) * 100 : null;
  return (
    <section className="panel evidence-panel" aria-label="Historical Model Performance">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Past performance</p>
          <h2>Price Model Historical Performance</h2>
        </div>
        {hasRatio && (
          <span className={`verdict ${beatBaseline ? 'positive' : 'caution'}`}>
            {beatBaseline ? 'More accurate than assuming no price change' : 'No better than assuming no price change'}
          </span>
        )}
      </div>
      <div className="metrics-grid">
        <article>
          <span>Average error</span>
          <strong className="mono">{formatPercent(backtest.mae_percent, 2)}</strong>
          <small className="kpi-subtext">Average size of the prediction mistakes</small>
        </article>
        <article>
          <span>Larger-error score</span>
          <strong className="mono">{formatPercent(backtest.rmse_percent, 2)}</strong>
          <small className="kpi-subtext">Gives bigger mistakes more weight; lower is better</small>
        </article>
        <article>
          <span>Up or down correct</span>
          <strong className="mono">{formatPercent(directionAcc, 1)}</strong>
          <small className="kpi-subtext">How often the price direction was right</small>
        </article>
        <article>
          <span>Compared with no price change</span>
          <strong className="mono">{Number.isFinite(ratio) ? `${ratio.toFixed(2)}×` : '—'}</strong>
          <small className="kpi-subtext">Below 1 means smaller average mistakes</small>
        </article>
      </div>
      <p className="method-note">
        Tested on {backtest.test_samples} past forecasts from {backtest.test_start} to {backtest.test_end}.
        These are retrospective point forecast results. The displayed range midpoint has not been separately evaluated. Past results do not guarantee future performance.
      </p>
    </section>
  );
}

function formatNewsTimestamp(publishedAt) {
  if (!publishedAt) return '';
  const raw = String(publishedAt).trim();
  const isDateOnly = /^\d{4}-\d{2}-\d{2}$/.test(raw);
  try {
    const d = new Date(isDateOnly ? `${raw}T00:00:00Z` : raw);
    if (Number.isNaN(d.getTime())) return raw.slice(0, 10);
    const datePart = d.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      timeZone: 'UTC',
    });
    if (isDateOnly) return datePart;
    return (
      datePart +
      ' · ' +
      d.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
        timeZone: 'UTC',
      }) +
      ' UTC'
    );
  } catch {
    return raw.slice(0, 10);
  }
}

export function isSecFilingItem(item) {
  if (!item || typeof item !== 'object') return false;
  if (item.kind === 'sec_filing' || item.type === 'sec_filing' || item.is_sec_filing === true) return true;
  if (typeof item.form === 'string' && item.form.trim()) return true;
  if (typeof item.filing_type === 'string' && item.filing_type.trim()) return true;
  const provider = String(item.provider || '').trim().toLowerCase();
  const source = String(item.source || '').trim().toLowerCase();
  return provider === 'sec_edgar' || source === 'sec edgar' || source === 'sec' || source.startsWith('sec ');
}

export function resolveSecFormLabel(item) {
  const rawForm = String(item?.form || item?.filing_type || '').trim().toUpperCase();
  if (rawForm) {
    return rawForm.startsWith('SEC') ? rawForm : `SEC ${rawForm}`;
  }
  const title = String(item?.title || item?.headline || '');
  const formMatch = title.match(/\b(10-K|10-Q|8-K|20-F|6-K|DEF\s*14A|S-1|S-3|13F-HR|SC\s*13[DG]|FORM\s*4)\b/i);
  if (formMatch) {
    const normalized = formMatch[1].replace(/\s+/g, ' ').toUpperCase().replace(/^FORM\s+/, '');
    return `SEC ${normalized}`;
  }
  return 'SEC Filing';
}

function resolveSentimentBadge(item) {
  const label = String(item.sentiment_label || '').toLowerCase();
  const badge = item.sentiment_badge;
  const score = item.sentiment;
  if (badge === 'Bullish' || label === 'positive' || label === 'bullish' || score > 0.15) {
    return {
      type: 'bullish',
      label: 'Positive tone',
    };
  }
  if (badge === 'Bearish' || label === 'negative' || label === 'bearish' || score < -0.15) {
    return {
      type: 'bearish',
      label: 'Negative tone',
    };
  }
  return {
    type: 'neutral',
    label: 'Neutral tone',
  };
}

function formatProviderLabel(provider) {
  if (!provider || provider === 'none') return null;
  const p = String(provider).toLowerCase();
  if (p === 'institutional_feed') return 'News feed';
  if (p === 'alpaca') return 'Alpaca';
  if (p === 'yahoo') return 'Yahoo Finance';
  if (p === 'sec_edgar') return 'Company filings';
  return provider.replace(/_/g, ' ');
}

function mergeNewsAndFilings(news) {
  const baseItems = Array.isArray(news?.items) ? news.items : [];
  const extraFilings = Array.isArray(news?.sec_filings)
    ? news.sec_filings.map((filing) => ({ ...filing, kind: filing.kind || 'sec_filing' }))
    : [];
  if (!extraFilings.length) return baseItems;
  const seen = new Set();
  const combined = [];
  for (const entry of [...baseItems, ...extraFilings]) {
    if (!entry) continue;
    const key = entry.id || entry.url || `${entry.title || entry.headline || ''}-${entry.published_at || entry.filing_date || ''}`;
    if (seen.has(key)) continue;
    seen.add(key);
    combined.push(entry);
  }
  return combined;
}

function NewsPanel({ news, ticker, loading }) {
  const [showAll, setShowAll] = useState(false);
  const [filterKind, setFilterKind] = useState('all'); // 'all' | 'headlines' | 'filings'
  const items = useMemo(() => mergeNewsAndFilings(news), [news]);
  const secCount = useMemo(() => items.filter(isSecFilingItem).length, [items]);
  const headlineCount = items.length - secCount;
  const effectiveFilter = secCount > 0 && headlineCount > 0 ? filterKind : 'all';
  const filteredItems = useMemo(() => {
    if (effectiveFilter === 'filings') return items.filter(isSecFilingItem);
    if (effectiveFilter === 'headlines') return items.filter((item) => !isSecFilingItem(item));
    return items;
  }, [items, effectiveFilter]);
  const providerLabel = formatProviderLabel(news?.provider);
  const displayedItems = showAll ? filteredItems : filteredItems.slice(0, 5);

  return (
    <section className="panel news-panel" aria-label="Live Market Headlines">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">In the news</p>
          <h2>Recent {ticker} headlines</h2>
        </div>
        <div className="heading-badges">
          {providerLabel && (
            <span className="source-provider-pill">
              <span className="provider-dot" aria-hidden="true" />
              {providerLabel}
            </span>
          )}
          <span className="context-label">
            <span className="context-dot" aria-hidden="true" />
            Not included in the forecast
          </span>
        </div>
      </div>

      <p className="method-note news-top-disclosure">
        These headlines and regulatory filings are provided for market context only and do not affect the displayed price forecast.
      </p>

      {secCount > 0 && headlineCount > 0 && (
        <div className="news-filter-bar" role="group" aria-label="Filter news and regulatory filings">
          <button
            type="button"
            className={`news-filter-btn ${filterKind === 'all' ? 'active' : ''}`}
            aria-pressed={filterKind === 'all'}
            onClick={() => setFilterKind('all')}
          >
            All ({items.length})
          </button>
          <button
            type="button"
            className={`news-filter-btn ${filterKind === 'headlines' ? 'active' : ''}`}
            aria-pressed={filterKind === 'headlines'}
            onClick={() => setFilterKind('headlines')}
          >
            Headlines ({headlineCount})
          </button>
          <button
            type="button"
            className={`news-filter-btn ${filterKind === 'filings' ? 'active' : ''}`}
            aria-pressed={filterKind === 'filings'}
            onClick={() => setFilterKind('filings')}
          >
            SEC Filings ({secCount})
          </button>
        </div>
      )}

      {loading && !items.length ? (
        <div className="news-loading-state" role="status" aria-label="Loading headlines">
          <div className="news-skeleton-pulse" />
          <span>Loading recent stories…</span>
        </div>
      ) : filteredItems.length ? (
        <>
          <ul className="news-simple-list">
            {displayedItems.map((item, index) => {
              const isFiling = isSecFilingItem(item);
              const filingLabel = isFiling ? resolveSecFormLabel(item) : null;
              const sentiment = resolveSentimentBadge(item);
              const title = item.title || item.headline || 'Market Update';
              const rawDate = item.published_at || item.filing_date || '';
              const timestamp = formatNewsTimestamp(rawDate);
              const sourceLabel = item.source || (isFiling ? 'SEC EDGAR' : 'Market Wire');
              return (
                <li
                  key={`${rawDate}-${item.id || index}`}
                  className={`news-list-item ${isFiling ? 'sec-filing-item' : ''}`}
                >
                  <div className="news-item-meta">
                    <span className="news-source">{sourceLabel}</span>
                    {timestamp && <time className="news-time" dateTime={rawDate}>{timestamp}</time>}
                    {isFiling ? (
                      <span className="sec-filing-badge">
                        {filingLabel}
                      </span>
                    ) : (
                      <span className={`sentiment-badge ${sentiment.type}`}>
                        {sentiment.label}
                      </span>
                    )}
                  </div>
                  <h3 className="news-item-headline">
                    {item.url ? (
                      <a href={item.url} target="_blank" rel="noreferrer noopener">
                        {title} <span className="external-arrow" aria-hidden="true">↗</span>
                      </a>
                    ) : (
                      title
                    )}
                  </h3>
                </li>
              );
            })}
          </ul>

          {filteredItems.length > 5 && (
            <button
              type="button"
              className="show-more-news-btn"
              onClick={() => setShowAll((prev) => !prev)}
            >
              {showAll ? 'Show fewer stories' : `Show more stories (${filteredItems.length - 5} more)`}
            </button>
          )}
        </>
      ) : (
        <p className="empty-copy">No recent stories are available right now.</p>
      )}
    </section>
  );
}

export default function App() {
  const initialDeepLink = useMemo(
    () => parseDeepLinkSearch(typeof window !== 'undefined' ? window.location.search : ''),
    [],
  );
  const [inputTicker, setInputTicker] = useState(initialDeepLink.ticker || '');
  const [submittedTicker, setSubmittedTicker] = useState('');
  const [chartTicker, setChartTicker] = useState(null);
  const [selectedRange, setSelectedRange] = useState(initialDeepLink.range);
  const [activeTab, setActiveTab] = useState(initialDeepLink.tab); // 'overview' | 'news' | 'performance'
  const [serviceStatus, setServiceStatus] = useState('checking');
  const [wakeAttempt, setWakeAttempt] = useState(1);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecast, setForecast] = useState(null);
  const [news, setNews] = useState(null);
  const [newsLoading, setNewsLoading] = useState(false);
  const [stockInfo, setStockInfo] = useState(null);
  const [inputError, setInputError] = useState('');
  const [historyError, setHistoryError] = useState('');
  const [forecastError, setForecastError] = useState('');
  const [refreshToken, setRefreshToken] = useState(0);
  const [marketHistory, setMarketHistory] = useState(null);
  const [perf, setPerf] = useState(null);

  const loading = historyLoading || forecastLoading;

  const wakeController = useRef(null);
  const requestController = useRef(null);
  const requestSeq = useRef(0);
  const perfT0 = useRef(0);
  const marketHistoryRef = useRef(marketHistory);
  marketHistoryRef.current = marketHistory;
  const submittedTickerRef = useRef(submittedTicker);
  submittedTickerRef.current = submittedTicker;
  const selectedRangeRef = useRef(selectedRange);
  selectedRangeRef.current = selectedRange;
  const activeTabRef = useRef(activeTab);
  activeTabRef.current = activeTab;

  const wake = useCallback(async () => {
    wakeController.current?.abort();
    const controller = new AbortController();
    wakeController.current = controller;
    setServiceStatus('checking');
    try {
      await wakeForecastService({ signal: controller.signal, onAttempt: setWakeAttempt });
      if (!controller.signal.aborted) setServiceStatus('online');
    } catch (wakeError) {
      if (wakeError?.name !== 'AbortError') setServiceStatus('offline');
    }
  }, []);

  useEffect(() => {
    wake();
    return () => wakeController.current?.abort();
  }, [wake]);

  useEffect(() => () => requestController.current?.abort(), []);

  const nowMs = useCallback(() => (typeof performance !== 'undefined' ? performance.now() : Date.now()), []);

  const handleHistorySettled = useCallback((report) => {
    const chartMs = Number.isFinite(report?.fetchMs)
      ? Math.round(report.fetchMs)
      : Math.round(nowMs() - (perfT0.current || nowMs()));
    const cacheLabel = !report?.ok
      ? 'history failed'
      : report.fromCache
        ? 'browser cache'
        : report.degraded
          ? 'forecast payload'
          : report.marketDataCache === 'hit'
            ? 'cache hit'
            : 'cold fetch';
    setPerf((prev) => ({ ...(prev || {}), chartMs, fetchMs: report?.fetchMs ?? null, cacheLabel, degraded: Boolean(report?.degraded) }));
  }, [nowMs]);

  const runForecast = useCallback(async (
    eventOrSymbol,
    { forceRefresh = false, skipPushState = false, initialRange, initialTab } = {},
  ) => {
    if (eventOrSymbol && typeof eventOrSymbol.preventDefault === 'function') {
      eventOrSymbol.preventDefault();
    }
    const raw = typeof eventOrSymbol === 'string' ? eventOrSymbol : inputTicker;
    const symbol = String(raw || '').trim().toUpperCase();
    if (!symbol) return;

    if (!/^[A-Z0-9.\-_]{1,15}$/.test(symbol)) {
      setInputError('Please enter a valid stock ticker symbol (e.g. MSFT, SHEL.L, NVDA, ARM).');
      return;
    }
    if (!ALL_TICKERS_SET.has(symbol)) {
      setInputError(`Choose one of the ${ALL_VALID_TICKERS.length} supported LSE, NASDAQ, and NYSE tickers. This symbol is not supported yet.`);
      return;
    }

    const nextRange = initialRange !== undefined ? initialRange : selectedRangeRef.current;
    const nextTab = initialTab !== undefined ? initialTab : activeTabRef.current;
    if (initialRange !== undefined) {
      setSelectedRange(initialRange);
      selectedRangeRef.current = initialRange;
    }
    if (initialTab !== undefined) {
      setActiveTab(initialTab);
      activeTabRef.current = initialTab;
    }

    setInputTicker(symbol);
    if (!skipPushState && !forceRefresh && typeof window !== 'undefined' && typeof window.history?.pushState === 'function') {
      const targetUrl = buildDeepLinkUrl({
        ticker: symbol,
        range: nextRange || '1M',
        tab: nextTab || 'overview',
      });
      if (`${window.location.pathname}${window.location.search}` !== targetUrl) {
        window.history.pushState(
          { ticker: symbol, range: nextRange || '1M', tab: nextTab || 'overview' },
          '',
          targetUrl,
        );
      }
    }

    if (forceRefresh) {
      setRefreshToken((value) => value + 1);
    }
    setInputError('');
    setHistoryError('');
    setForecastError('');

    // Cancel outstanding requests and ignore late responses
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    const seq = ++requestSeq.current;

    const isSameTicker = chartTicker === symbol;
    // Clear previous results when selecting a different stock
    if (!isSameTicker) {
      setChartTicker(null);
      setMarketHistory(null);
      setStockInfo(null);
    }
    setForecast(null);
    setNews(null);
    setPerf(null);
    setSubmittedTicker(symbol);
    submittedTickerRef.current = symbol;
    setHistoryLoading(true);
    setForecastLoading(true);
    setNewsLoading(true);

    perfT0.current = (typeof performance !== 'undefined' ? performance.now() : Date.now());

    let historyUsable = isSameTicker && Boolean(marketHistoryRef.current?.daily?.length);
    let fallbackAvailable = false;
    let historyFailureMessage = '';

    // 1. History request: decoupled from slow /forecast, reveals chart immediately when ready
    const historyPromise = fetchPriceHistory(symbol, { signal: controller.signal, forceRefresh })
      .then((historyResult) => {
        if (requestSeq.current !== seq) return null;
        assertTickerIdentity(historyResult, symbol);
        setMarketHistory(historyResult);
        if (Array.isArray(historyResult?.daily) && historyResult.daily.length > 0) {
          historyUsable = true;
          setChartTicker(symbol);
        }
        setHistoryLoading(false);
        return historyResult;
      })
      .catch((err) => {
        if (requestSeq.current !== seq) return null;
        setHistoryLoading(false);
        if (err?.name !== 'AbortError') {
          historyFailureMessage = err?.message || 'Price history is unavailable for this stock right now.';
          return { error: err };
        }
        return null;
      });

    // 2. Forecast request
    const forecastPromise = (async () => {
      if (serviceStatus !== 'online') {
        await wakeForecastService({ signal: controller.signal, onAttempt: setWakeAttempt });
        if (requestSeq.current !== seq) return null;
        setServiceStatus('online');
      }
      const forecastT0 = nowMs();
      const forecastValue = await fetchSimpleForecast(symbol, { signal: controller.signal, forceRefresh });
      if (requestSeq.current !== seq) return null;
      assertTickerIdentity(forecastValue, symbol);
      setForecast(forecastValue);
      setPerf((prev) => ({ ...(prev || {}), forecastMs: Math.round(nowMs() - forecastT0) }));

      // Fallback check if primary history failed
      const dates = forecastValue?.historical_dates;
      const prices = forecastValue?.historical_prices;
      if (
        forecastValue?.historical_provenance !== 'synthetic' &&
        Array.isArray(dates) &&
        Array.isArray(prices) &&
        dates.length >= 2
      ) {
        fallbackAvailable = true;
        if (!historyUsable) {
          setChartTicker(symbol);
        }
      }
      setForecastLoading(false);
      return forecastValue;
    })().catch((err) => {
      if (requestSeq.current !== seq) return null;
      setForecastLoading(false);
      if (err?.name !== 'AbortError') {
        setForecastError(err?.message || 'The forecast could not be completed.');
      }
      return null;
    });

    // 3. News request
    const newsPromise = fetchTickerNews(symbol, { signal: controller.signal, forceRefresh })
      .then((newsResult) => {
        if (requestSeq.current !== seq) return null;
        assertTickerIdentity(newsResult, symbol);
        setNews(newsResult);
        setNewsLoading(false);
        return newsResult;
      })
      .catch((err) => {
        if (requestSeq.current !== seq) return null;
        setNewsLoading(false);
        if (err?.name !== 'AbortError') {
          setNews({ status: 'unavailable', items: [] });
        }
        return null;
      });

    // 4. Lazy non-blocking company fundamentals request (/api/v1/info)
    const infoPromise = fetchStockInfo(symbol, { signal: controller.signal, forceRefresh })
      .then((infoResult) => {
        if (requestSeq.current !== seq) return null;
        setStockInfo(infoResult);
        return infoResult;
      })
      .catch(() => null);

    await Promise.allSettled([historyPromise, forecastPromise]);
    if (requestSeq.current !== seq) return;
    setHistoryLoading(false);
    setForecastLoading(false);

    if (!historyUsable && !fallbackAvailable) {
      setHistoryError(historyFailureMessage || 'Price history is unavailable for this stock right now.');
    }
    await Promise.allSettled([newsPromise, infoPromise]);
  }, [inputTicker, nowMs, serviceStatus, chartTicker]);

  const runForecastRef = useRef(runForecast);
  runForecastRef.current = runForecast;

  // Hydrate initial URL deep link on mount (?ticker=MSFT&range=1M&tab=overview)
  useEffect(() => {
    if (initialDeepLink.ticker) {
      runForecastRef.current(initialDeepLink.ticker, {
        skipPushState: true,
        initialRange: initialDeepLink.range,
        initialTab: initialDeepLink.tab,
      });
    }
  }, [initialDeepLink]);

  // Browser Back / Forward navigation via popstate
  useEffect(() => {
    if (typeof window === 'undefined') return undefined;
    const onPopState = () => {
      const parsed = parseDeepLinkSearch(window.location.search);
      setActiveTab(parsed.tab);
      activeTabRef.current = parsed.tab;
      setSelectedRange(parsed.range);
      selectedRangeRef.current = parsed.range;

      if (parsed.ticker) {
        if (parsed.ticker !== submittedTickerRef.current) {
          runForecastRef.current(parsed.ticker, {
            skipPushState: true,
            initialRange: parsed.range,
            initialTab: parsed.tab,
          });
        } else {
          setInputTicker(parsed.ticker);
        }
      } else {
        requestController.current?.abort();
        requestSeq.current += 1;
        setInputTicker('');
        setSubmittedTicker('');
        submittedTickerRef.current = '';
        setChartTicker(null);
        setMarketHistory(null);
        setForecast(null);
        setNews(null);
        setStockInfo(null);
        setPerf(null);
        setInputError('');
        setHistoryError('');
        setForecastError('');
        setHistoryLoading(false);
        setForecastLoading(false);
        setNewsLoading(false);
      }
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  const selectTab = useCallback((nextTab) => {
    setActiveTab(nextTab);
    activeTabRef.current = nextTab;
    const currentSymbol = submittedTickerRef.current;
    if (currentSymbol && typeof window !== 'undefined' && typeof window.history?.replaceState === 'function') {
      const targetUrl = buildDeepLinkUrl({
        ticker: currentSymbol,
        range: selectedRangeRef.current || '1M',
        tab: nextTab,
      });
      window.history.replaceState(
        { ticker: currentSymbol, range: selectedRangeRef.current || '1M', tab: nextTab },
        '',
        targetUrl,
      );
    }
  }, []);

  const handleRangeChange = useCallback((nextRange) => {
    setSelectedRange(nextRange);
    selectedRangeRef.current = nextRange;
    const currentSymbol = submittedTickerRef.current;
    if (currentSymbol && nextRange && typeof window !== 'undefined' && typeof window.history?.replaceState === 'function') {
      const targetUrl = buildDeepLinkUrl({
        ticker: currentSymbol,
        range: nextRange,
        tab: activeTabRef.current || 'overview',
      });
      window.history.replaceState(
        { ticker: currentSymbol, range: nextRange, tab: activeTabRef.current || 'overview' },
        '',
        targetUrl,
      );
    }
  }, []);

  const handleRetry = useCallback(() => {
    if (submittedTicker) {
      runForecast(submittedTicker, { forceRefresh: true, skipPushState: true });
    }
  }, [runForecast, submittedTicker]);

  let identityError = '';
  try { if (forecast && marketHistory) assertMarketIdentity(forecast, marketHistory); }
  catch (error) { identityError = error.message; }

  const unifiedError = useMemo(() => {
    if (loading) return null;
    if (!historyError && !forecastError) return null;
    if (historyError && forecastError) {
      const same = historyError.trim().toLowerCase() === forecastError.trim().toLowerCase();
      return {
        message: same ? historyError : `${historyError} ${forecastError}`,
        actionLabel: 'Retry',
      };
    }
    if (historyError) {
      return { message: historyError, actionLabel: 'Retry' };
    }
    return { message: forecastError, actionLabel: 'Retry forecast' };
  }, [loading, historyError, forecastError]);

  const resolvedCurrencySymbol = chartTicker
    ? (
      forecast?.ticker === chartTicker && forecast?.currency_symbol
        ? forecast.currency_symbol
        : (marketHistory?.ticker === chartTicker && marketHistory?.currencySymbol
          ? marketHistory.currencySymbol
          : (stockInfo?.ticker === chartTicker && stockInfo?.currencySymbol
            ? stockInfo.currencySymbol
            : (chartTicker.endsWith('.L') ? 'p' : '$')))
    )
    : '$';

  const resolvedCompanyName = chartTicker
    ? (
      (forecast?.ticker === chartTicker && forecast?.ticker_name)
      || (stockInfo?.ticker === chartTicker && stockInfo?.name)
      || null
    )
    : null;

  const totalNewsCount = useMemo(() => mergeNewsAndFilings(news).length, [news]);

  return (
    <div className="app-shell">
      <div className="ambient-orbs" aria-hidden="true">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
      </div>

      <header className="topbar">
        <a className="brand" href="#top" aria-label="Signal Seven home">
          <span className="brand-icon">S7</span>
          <div className="brand-title-wrap">
            <b>Signal Seven</b>
            <small>Stock forecasts made simple</small>
          </div>
        </a>

        <div className="topbar-right">
          <ServiceBadge status={serviceStatus} attempt={wakeAttempt} onRetry={wake} />
        </div>
      </header>

      <main id="top">
        {/* Search Header: Large heading on empty first visit; compact row beneath brand after submission */}
        <section className={`search-section ${submittedTicker ? 'compact-search-section' : 'opening-search-section'}`}>
          {!submittedTicker && (
            <h1 className="explore-heading">Which stock would you like to explore?</h1>
          )}

          <form className="stock-search-form" onSubmit={runForecast} noValidate>
            <div className="search-input-row">
              <div className="input-field-group">
                <label htmlFor="tickerInput" className="sr-only">Stock ticker</label>
                <StockAutocomplete
                  value={inputTicker}
                  onChange={(value) => { setInputTicker(value); if (inputError) setInputError(''); }}
                  onSelect={(symbol) => runForecast(symbol)}
                  invalid={Boolean(inputError)}
                  errorId={inputError ? 'symbolError' : undefined}
                />
                {inputError && (
                  <span id="symbolError" className="symbol-feedback" role="alert">
                    {inputError}
                  </span>
                )}
              </div>

              <button
                className="view-outlook-btn"
                type="submit"
                disabled={!inputTicker.trim()}
              >
                View outlook
              </button>
            </div>
          </form>

          {loading && (
            <div className="search-progress" role="status" aria-live="polite">
              <span className="progress-dot" aria-hidden="true" />
              <span>
                {serviceStatus !== 'online'
                  ? `Starting forecast service${wakeAttempt > 1 ? ` · attempt ${wakeAttempt}` : ''}… Preparing data for ${submittedTicker}…`
                  : historyLoading
                    ? `Loading data for ${submittedTicker}…`
                    : `Loading forecast for ${submittedTicker}…`}
              </span>
            </div>
          )}

          {unifiedError && (
            <div className="actionable-forecast-error" role="alert">
              <span>{unifiedError.message}</span>
              <button
                type="button"
                className="retry-action-btn"
                onClick={handleRetry}
              >
                {unifiedError.actionLabel}
              </button>
            </div>
          )}
        </section>

        {/* Results Workspace */}
        {chartTicker && (
          <div className="results">
            {/* Centerpiece Interactive Price Chart */}
            <PriceChart
              ticker={chartTicker}
              refreshToken={refreshToken}
              marketHistory={marketHistory}
              historyLoading={historyLoading}
              historyError={historyError}
              onRetryHistory={handleRetry}
              selectedRange={selectedRange}
              onRangeChange={handleRangeChange}
              currencySymbol={resolvedCurrencySymbol}
              forecast={forecast?.ticker === chartTicker ? forecast : null}
              companyName={resolvedCompanyName}
              onHistorySettled={handleHistorySettled}
              onRetryForecast={handleRetry}
            />

            {/* Supporting Information Behind Overview / News / Performance Tabs */}
            <div className="supporting-tabs-container">
              <div
                className="tab-navigation-bar"
                role="tablist"
                aria-label="Supporting information tabs"
                onKeyDown={(event) => {
                  const keys = ['ArrowRight', 'ArrowLeft', 'Home', 'End'];
                  if (!keys.includes(event.key)) return;
                  event.preventDefault();
                  const tabs = ['overview', 'news', 'performance'];
                  const index = tabs.indexOf(activeTab);
                  const next = event.key === 'Home' ? 0 : event.key === 'End' ? 2
                    : (index + (event.key === 'ArrowRight' ? 1 : 2)) % 3;
                  selectTab(tabs[next]);
                  document.getElementById(`tab-${tabs[next]}`)?.focus();
                }}
              >
                <button
                  type="button"
                  role="tab"
                  id="tab-overview"
                  tabIndex={activeTab === 'overview' ? 0 : -1}
                  aria-selected={activeTab === 'overview'}
                  aria-controls="panel-overview"
                  className={`sub-tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
                  onClick={() => selectTab('overview')}
                >
                  Overview
                </button>
                <button
                  type="button"
                  role="tab"
                  id="tab-news"
                  tabIndex={activeTab === 'news' ? 0 : -1}
                  aria-selected={activeTab === 'news'}
                  aria-controls="panel-news"
                  className={`sub-tab-btn ${activeTab === 'news' ? 'active' : ''}`}
                  onClick={() => selectTab('news')}
                >
                  News{totalNewsCount ? ` (${totalNewsCount})` : ''}
                </button>
                <button
                  type="button"
                  role="tab"
                  id="tab-performance"
                  tabIndex={activeTab === 'performance' ? 0 : -1}
                  aria-selected={activeTab === 'performance'}
                  aria-controls="panel-performance"
                  className={`sub-tab-btn ${activeTab === 'performance' ? 'active' : ''}`}
                  onClick={() => selectTab('performance')}
                >
                  Performance
                </button>
              </div>

              <div id="panel-overview" role="tabpanel" aria-labelledby="tab-overview" hidden={activeTab !== 'overview'}>
                <CompanyFundamentalsStrip
                  info={stockInfo}
                  ticker={chartTicker}
                  currencySymbol={resolvedCurrencySymbol}
                />
                <VolatilityOutlook
                  refreshToken={refreshToken}
                  expectedIdentity={marketHistory || forecast}
                  ticker={chartTicker}
                  currencySymbol={resolvedCurrencySymbol}
                  currentPrice={forecast?.ticker === chartTicker ? forecast?.current_price : null}
                />
              </div>

              <div id="panel-news" role="tabpanel" aria-labelledby="tab-news" hidden={activeTab !== 'news'}>
                <NewsPanel
                  key={chartTicker}
                  news={news}
                  ticker={chartTicker}
                  loading={newsLoading}
                />
              </div>

              <div id="panel-performance" role="tabpanel" aria-labelledby="tab-performance" hidden={activeTab !== 'performance'}>
                {forecast?.backtest && !identityError ? (
                  <BacktestPanel backtest={forecast.backtest} />
                ) : (
                  <div className="panel empty-performance-panel">
                    <p className="empty-copy">{identityError || (forecastLoading ? 'Loading price evidence…' : 'Price evaluation is unavailable for this forecast. Retry the forecast to try again.')}</p>
                  </div>
                )}

                <ForecastLedgerTrackRecord ticker={chartTicker} enabled={activeTab === 'performance'} />

                {perf && (
                  <details className="perf-technical-details">
                    <summary>Technical & operational diagnostics</summary>
                    <dl className="technical-dl">
                      <div>
                        <dt>Price history load:</dt>
                        <dd className="mono">{perf.chartMs != null ? `${perf.chartMs}ms` : '—'}</dd>
                      </div>
                      <div>
                        <dt>Forecast compute:</dt>
                        <dd className="mono">{perf.forecastMs != null ? `${perf.forecastMs}ms` : '—'}</dd>
                      </div>
                      <div>
                        <dt>Market data cache:</dt>
                        <dd className="mono">{perf.cacheLabel || '—'}</dd>
                      </div>
                    </dl>
                  </details>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      <footer>
        <span>Experimental estimates, not financial advice.</span>
      </footer>
    </div>
  );
}
