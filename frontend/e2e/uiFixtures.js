const origin = '2026-09-03';
const futureDates = ['2026-09-04', '2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11', '2026-09-14', '2026-09-15'];
const volatilityFutureDates = [...futureDates, '2026-09-16', '2026-09-17', '2026-09-18',
  '2026-09-21', '2026-09-22', '2026-09-23', '2026-09-24', '2026-09-25',
  '2026-09-28', '2026-09-29', '2026-09-30', '2026-10-01', '2026-10-02'];
const historyDates = ['2026-08-25', '2026-08-26', '2026-08-27', '2026-08-28',
  '2026-08-31', '2026-09-01', '2026-09-02', origin];

export function volatilityPayload(horizon, model = 'gpu_g3') {
  const learned = model === 'gpu_g3';
  const dates = volatilityFutureDates.slice(0, horizon);
  const historical = historyDates;
  const quantiles = Object.fromEntries([
    ['p05', 435], ['p10', 440], ['p25', 445], ['p50', 450],
    ['p75', 455], ['p90', 460], ['p95', 465],
  ].map(([key, value]) => [key, Array(horizon).fill(value)]));
  return {
    ticker: 'MSFT', as_of: origin, horizon, current_price: 450,
    historical_dates: historical, historical_prices: historical.map((_, i) => 447.9 + i * 0.3),
    forecast: {
      future_dates: dates,
      price_quantiles: quantiles,
      expected_annualized_volatility: 0.214,
      predicted_volatility: 0.214,
      model, requested_model: learned ? model : 'auto', baseline: !learned,
    },
    evidence: {
      model_status: learned ? 'learned_model' : 'baseline', baseline: !learned, model_name: model,
      ...(learned
        ? { model_version: 'g3-qlike-base-margin-v2', metric_source: 'validation_panel' }
        : { model_family: 'statistical_baseline', default_policy: 'v5_duan_har' }),
      data_as_of: origin, data_provider: 'local_fixture', risk_level: 'Moderate',
      trailing_annualized_volatility_60d: 0.19, fallback_used: null,
    },
  };
}

export function volatilityBatchPayload(horizons = [5, 10, 20], failedHorizon = null, ticker = 'MSFT', currentPrice = 450, model = 'gpu_g3') {
  const forecasts = {};
  const errors = {};
  for (const horizon of horizons) {
    if (horizon === failedHorizon) {
      errors[String(horizon)] = { code: 'HORIZON_UNAVAILABLE', message: 'Fixture horizon unavailable' };
    } else {
      const entry = volatilityPayload(horizon, model);
      entry.ticker = ticker;
      entry.current_price = currentPrice;
      if (currentPrice !== 450) {
        for (const key of Object.keys(entry.forecast.price_quantiles)) {
          entry.forecast.price_quantiles[key] = Array(horizon).fill(currentPrice);
        }
      }
      forecasts[String(horizon)] = entry;
    }
  }
  return {
    ticker,
    as_of: origin,
    requested_model: 'auto',
    horizons,
    forecasts,
    errors,
    post_close_grace_used: false,
  };
}

export function installFixtures(page, failedHorizon = null, priceOverrides = {}, volatilityModel = 'gpu_g3') {
  return page.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith('/health')) return route.fulfill({ json: { status: 'ok' } });
    if (!url.pathname.startsWith('/api/')) return route.continue();
    if (url.pathname.endsWith('/history')) {
      return route.fulfill({ json: {
        ticker: 'MSFT', as_of: origin,
        intraday_available: true,
        daily: historyDates.map((d, i) => ({ d, c: 447.9 + i * 0.3 })),
      } });
    }
    if (url.pathname.endsWith('/volatility/forecast/batch')) {
      const horizons = (url.searchParams.get('horizons') || '5,10,20').split(',').map(Number);
      return route.fulfill({ json: volatilityBatchPayload(horizons, failedHorizon, 'MSFT', 450, volatilityModel) });
    }
    if (url.pathname.includes('/volatility/forecast')) {
      const horizon = Number(url.searchParams.get('horizon') || 5);
      if (horizon === failedHorizon) {
        return route.fulfill({ status: 503, json: { detail: 'Fixture horizon unavailable' } });
      }
      return route.fulfill({ json: volatilityPayload(horizon, volatilityModel) });
    }
    if (url.pathname.endsWith('/forecast')) {
      return route.fulfill({ json: {
        ticker: 'MSFT', forecast_days: 7, data_as_of: origin, current_price: 450,
        future_dates: futureDates,
        lower_prices: [440, 441, 442, 443, 444, 445, 446],
        upper_prices: [460, 461, 462, 463, 464, 465, 466],
        ...priceOverrides,
      } });
    }
    if (url.pathname.endsWith('/volatility/ledger')) {
      return route.fulfill({ json: {
        ticker: 'MSFT',
        entries: [],
        live_track_record: { total_forecasts: 0 },
        replay_track_record: { total_forecasts: 0 },
      } });
    }
    if (url.pathname.endsWith('/info')) {
      return route.fulfill({ json: {
        ticker: 'MSFT',
        name: 'Microsoft Corp.',
        sector: 'Information Technology',
        currency: 'USD',
        marketCap: 3_120_000_000_000,
        peRatio: null,
        fiftyTwoWeekLow: 385.5,
        fiftyTwoWeekHigh: 468.35,
      } });
    }
    if (url.pathname.endsWith('/intraday')) {
      const bars = Array.from({ length: 6 }, (_, i) => ({
        t: `${origin}T14:0${i}:00Z`,
        c: 448.5 + i * 0.3,
      }));
      return route.fulfill({ json: {
        ticker: 'MSFT',
        session: origin,
        bars,
        bar_count: bars.length,
        available: true,
      } });
    }
    if (url.pathname.endsWith('/news')) return route.fulfill({ json: {
      status: 'available', provider: 'verification_fixture', ticker: 'MSFT',
      items: [{ id: 'example-1', title: '[Example fixture] Company update for layout verification',
        source: 'Example fixture — not live news', published_at: '2026-09-03T12:00:00Z' }],
      sec_filings: [{
        id: 'sec-10k-1',
        kind: 'sec_filing',
        form: '10-K',
        title: 'Annual Report (Form 10-K)',
        source: 'SEC EDGAR',
        filing_date: '2026-08-01',
        url: 'https://www.sec.gov/example-10k',
      }],
    } });
    return route.fulfill({ json: {} });
  });
}
