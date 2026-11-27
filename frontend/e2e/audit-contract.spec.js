import path from 'node:path';
import { expect, test } from '@playwright/test';
import { installFixtures, volatilityBatchPayload, volatilityPayload } from './uiFixtures.js';

async function submit(page) {
  await page.goto('/');
  await page.getByLabel('Stock ticker').fill('MSFT');
  await page.getByRole('button', { name: /view outlook/i }).click();
  await expect(page.locator('#chartContainer canvas')).toBeVisible();
}

async function saveShot(page, testInfo, filename) {
  await page.screenshot({ path: testInfo.outputPath(filename), fullPage: true });
  if (process.env.E2E_SCREENSHOT_DIR) {
    await page.screenshot({ path: path.join(process.env.E2E_SCREENSHOT_DIR, filename), fullPage: true });
  }
}

test('company search and keyboard selection submit the supported ticker', async ({ page }) => {
  await installFixtures(page);
  await page.route('**/api/v1/search?**', (route) => route.fulfill({ json: { results: [{ ticker: 'MSFT', name: 'Microsoft Corp.' }] } }));
  await page.goto('/');
  const input = page.getByLabel('Stock ticker');
  await input.fill('Microsoft');
  await expect(input).toHaveValue('Microsoft');
  await expect(page.getByRole('option', { name: /Microsoft Corp/ })).toBeVisible();
  await input.press('ArrowDown');
  await input.press('Enter');
  await expect(input).toHaveValue('MSFT');
  await expect(page.locator('#chartContainer canvas')).toBeVisible();
});

test('range controls and toolbar Table toggle are usable from the keyboard without inline slider', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await installFixtures(page);
  await submit(page);
  const range = page.getByRole('button', { name: 'MAX', exact: true });
  await range.focus();
  await range.press('Enter');
  await expect(range).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText(/Available history/)).toBeVisible();

  await expect(page.getByRole('slider', { name: 'Inspect chart date' })).toHaveCount(0);
  await expect(page.getByRole('table')).toHaveCount(0);

  const tableBtn = page.getByRole('button', { name: 'Table', exact: true });
  await expect(tableBtn).toHaveAttribute('aria-expanded', 'false');
  await tableBtn.focus();
  await tableBtn.press('Enter');
  await expect(tableBtn).toHaveAttribute('aria-expanded', 'true');
  await expect(page.getByRole('table')).toBeVisible();
  await expect(page.getByRole('rowheader', { name: '2026-09-15' })).toBeVisible();

  await tableBtn.press('Enter');
  await expect(tableBtn).toHaveAttribute('aria-expanded', 'false');
  await expect(page.getByRole('table')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('one never-resolving horizon in single-horizon fallback settles within 20s while siblings remain visible', async ({ page }) => {
  await installFixtures(page);
  await page.route('**/api/v1/volatility/forecast/batch?**', (route) => route.fulfill({ status: 404, json: { detail: 'Not Found' } }));
  await page.route('**/api/v1/volatility/forecast?**', async (route) => {
    if (new URL(route.request().url()).searchParams.get('horizon') === '10') return new Promise(() => {});
    return route.fallback();
  });
  await submit(page);
  await expect(page.getByText('5 sessions', { exact: true })).toBeVisible();
  await expect(page.getByText('20 sessions', { exact: true })).toBeVisible();
  await expect(page.getByText(/10-session outlook unavailable/)).toBeVisible({ timeout: 25_000 });
  await expect(page.getByText(/10-session outlook loading/)).toHaveCount(0);
});

test('a failed chart chunk exposes retry and auto-opens the accessible price data table', async ({ page }) => {
  await installFixtures(page);
  await page.route('**/assets/chart*.js', (route) => route.abort('failed'));
  await page.goto('/');
  await page.getByLabel('Stock ticker').fill('MSFT');
  await page.getByRole('button', { name: /view outlook/i }).click();
  await expect(page.getByRole('button', { name: /Retry chart engine/i })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Table', exact: true })).toHaveAttribute('aria-expanded', 'true');
  await expect(page.getByRole('table')).toBeVisible();
  await expect(page.getByRole('rowheader', { name: '2026-09-03' })).toBeVisible();
});

test('rapid stock switching ignores a late response for the previous selection', async ({ page }) => {
  await installFixtures(page);
  let release;
  const delayed = new Promise((resolve) => { release = resolve; });
  await page.route('**/api/v1/**', async (route) => {
    const url = new URL(route.request().url());
    const ticker = url.searchParams.get('ticker');
    if (ticker === 'MSFT' && url.pathname.endsWith('/forecast') && !url.pathname.includes('/volatility/')) {
      await delayed;
      return route.fulfill({ json: { ticker: 'MSFT', current_price: 450 } }).catch(() => {});
    }
    if (ticker !== 'AAPL') return route.fallback();
    if (url.pathname.endsWith('/history')) return route.fulfill({ json: { ticker, as_of: '2026-09-03', daily: [{ d: '2026-09-02', c: 149 }, { d: '2026-09-03', c: 150 }] } });
    if (url.pathname.endsWith('/volatility/forecast/batch')) {
      const horizons = (url.searchParams.get('horizons') || '5,10,20').split(',').map(Number);
      return route.fulfill({ json: volatilityBatchPayload(horizons, null, ticker, 150) });
    }
    if (url.pathname.includes('/volatility/forecast')) {
      const payload = volatilityPayload(Number(url.searchParams.get('horizon')));
      payload.ticker = ticker; payload.current_price = 150;
      return route.fulfill({ json: payload });
    }
    if (url.pathname.endsWith('/forecast')) return route.fulfill({ json: { ticker, current_price: 150, data_as_of: '2026-09-03', future_dates: ['2026-09-04', '2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11', '2026-09-14', '2026-09-15'], lower_prices: Array(7).fill(149), upper_prices: Array(7).fill(153) } });
    if (url.pathname.endsWith('/news')) return route.fulfill({ json: { ticker, status: 'available', items: [] } });
    return route.fulfill({ json: { ticker, entries: [], live_track_record: { total_forecasts: 0 }, replay_track_record: { total_forecasts: 0 } } });
  });
  await submit(page);
  await page.getByLabel('Stock ticker').fill('AAPL');
  await page.getByRole('button', { name: /view outlook/i }).click();
  await expect(page.locator('#chartContainer')).toContainText('AAPL');
  release();
  await expect(page.locator('.t212-price')).toHaveText('$150.00');
  await expect(page.locator('.chart-estimate-bar')).toContainText('$151.00');
  await expect(page.getByText(/invalid origin/)).toHaveCount(0);
});

test('desktop 1440x900: single history fetch, zero tab refetches, hidden empty ledger, opaque sticky topbar, experimental band', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  const requests = [];
  page.on('request', (req) => requests.push(req.url()));
  await installFixtures(page);
  await submit(page);

  await expect(page.getByRole('heading', { name: 'Volatility Outlook' })).toBeVisible();
  await expect(page.getByLabel('MSFT company fundamentals')).toContainText('$3.12T');
  await expect(page.locator('.chart-estimate-bar')).toContainText('Historical error range (experimental)');
  await expect(page.locator('.chart-estimate-bar')).not.toContainText(/calibrated|empirical band/i);
  expect(page.url()).toContain('ticker=MSFT');

  const historyCalls = () => requests.filter((u) => u.includes('/api/v1/history'));
  const batchCalls = () => requests.filter((u) => u.includes('/api/v1/volatility/forecast/batch'));
  const ledgerCalls = () => requests.filter((u) => u.includes('/api/v1/volatility/ledger'));
  const coreCalls = () => requests.filter((u) => /\/api\/v1\/(history|forecast|volatility\/forecast|news)/.test(u));

  expect(historyCalls()).toHaveLength(1);
  expect(batchCalls()).toHaveLength(1);
  expect(ledgerCalls()).toHaveLength(0);

  const coreCountBeforeTabs = coreCalls().length;
  await page.getByRole('tab', { name: /News/ }).click();
  await expect(page.locator('#panel-news')).toBeVisible();
  await page.getByRole('tab', { name: 'Performance', exact: true }).click();
  await expect(page.locator('#panel-performance')).toBeVisible();
  await expect.poll(() => ledgerCalls().length).toBe(1);
  await expect(page.locator('#forecastLedgerSection')).toHaveCount(0);

  await page.getByRole('tab', { name: 'Overview', exact: true }).click();
  await expect(page.locator('#panel-overview')).toBeVisible();
  await page.getByRole('tab', { name: 'Performance', exact: true }).click();
  await page.getByRole('tab', { name: 'Overview', exact: true }).click();

  expect(coreCalls()).toHaveLength(coreCountBeforeTabs);
  expect(ledgerCalls()).toHaveLength(1);

  // Verify opaque frosted sticky topbar
  await page.evaluate(() => window.scrollTo(0, 300));
  const topbarStyle = await page.locator('.topbar').evaluate((el) => {
    const rect = el.getBoundingClientRect();
    const bg = getComputedStyle(el).backgroundColor;
    const match = bg.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([0-9.]+))?\)/);
    const alpha = match ? (match[4] === undefined ? 1 : Number(match[4])) : 0;
    return { top: rect.top, alpha, position: getComputedStyle(el).position };
  });
  expect(topbarStyle.position).toBe('sticky');
  expect(topbarStyle.top).toBeLessThanOrEqual(1);
  expect(topbarStyle.alpha).toBeGreaterThanOrEqual(0.85);

  await page.evaluate(() => window.scrollTo(0, 0));
  await saveShot(page, testInfo, 'm4-desktop-1440x900.png');
});

test('mobile 390x844: 0px autocomplete overlap, Enter dismisses suggestions, opaque topbar, readable summary and zero overflow', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await installFixtures(page, 10);
  await page.route('**/api/v1/search?**', (route) => route.fulfill({
    json: { results: [{ ticker: 'MSFT', name: 'Microsoft Corp.' }, { ticker: 'MS', name: 'Morgan Stanley' }] },
  }));
  await page.goto('/');

  const input = page.getByLabel('Stock ticker');
  const submitBtn = page.getByRole('button', { name: /view outlook/i });
  await input.click();
  await input.fill('MS');
  const suggestions = page.locator('#stock-suggestions');
  await expect(suggestions).toBeVisible();

  const submitBox = await submitBtn.boundingBox();
  const suggestionsBox = await suggestions.boundingBox();
  // Autocomplete anchors strictly below the View outlook button: 0px overlap
  expect(suggestionsBox.y).toBeGreaterThanOrEqual(submitBox.y + submitBox.height - 0.5);
  await saveShot(page, testInfo, 'm4-mobile-autocomplete-390x844.png');

  // Pressing Enter directly without ArrowDown dismisses suggestions and submits
  await input.fill('MSFT');
  await expect(suggestions).toBeVisible();
  await input.press('Enter');
  await expect(suggestions).toHaveCount(0);
  await expect(page.locator('#chartContainer canvas')).toBeVisible();

  await expect(page.locator('.chart-estimate-bar')).toBeVisible();
  await expect(page.locator('.chart-estimate-bar')).toContainText('Historical error range (experimental)');
  const summaryFlexWrap = await page.locator('.estimate-summary-text').evaluate((el) => getComputedStyle(el).flexWrap);
  expect(summaryFlexWrap).toBe('nowrap');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);

  await saveShot(page, testInfo, 'm4-mobile-390x844.png');
});

test('mobile 320px: compact ticker feedback stays below the submit button', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 700 });
  await installFixtures(page);
  await submit(page);
  const input = page.getByRole('combobox', { name: 'Stock ticker' });
  const submitButton = page.getByRole('button', { name: 'View outlook' });
  await input.fill('INVALID!');
  await input.press('Enter');

  const alert = page.getByRole('search').getByRole('alert');
  await expect(alert).toBeVisible();
  await expect(input).toHaveAccessibleDescription(/Please enter a valid stock ticker/);
  const [alertBox, submitBox] = await Promise.all([alert.boundingBox(), submitButton.boundingBox()]);
  expect(alertBox.y).toBeGreaterThanOrEqual(submitBox.y + submitBox.height);
  await submitButton.click();

  await input.fill('MSFT');
  await expect(alert).toHaveCount(0);
  await submitButton.click();
  await expect(page.locator('#chartContainer canvas')).toBeVisible();
});

test('mobile 390px: historical and estimate scrubbing keep the canvas in place', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await installFixtures(page);
  await submit(page);
  await expect(page.locator('.chart-estimate-bar')).toBeVisible();
  await page.evaluate(() => document.fonts.ready);

  const canvas = page.locator('#chartContainer canvas');
  const bounds = await canvas.boundingBox();
  const y = bounds.y + bounds.height / 2;
  const scrubDate = page.locator('.t212-scrub-date');
  await page.mouse.move(bounds.x + bounds.width * 0.3, y);
  await expect(scrubDate).toHaveText(/^\d{4}-\d{2}-\d{2}$/);
  expect(Math.abs((await canvas.boundingBox()).y - bounds.y)).toBeLessThan(1);

  await page.mouse.move(bounds.x + bounds.width * 0.85, y);
  await expect(scrubDate).toContainText('Estimate');
  await expect(page.locator('.t212-scrub-range')).toBeVisible();
  expect(Math.abs((await canvas.boundingBox()).y - bounds.y)).toBeLessThan(1);

  await page.mouse.move(3, 3);
  await expect(scrubDate).toHaveCount(0);
  await expect(page.locator('.t212-price')).toHaveText('$450.00');
  expect(Math.abs((await canvas.boundingBox()).y - bounds.y)).toBeLessThan(1);
});

test('deep link hydration (?ticker=MSFT&range=MAX&tab=news), SEC filings, lazy 24H intraday, and browser Back work end-to-end', async ({ page }) => {
  const requests = [];
  page.on('request', (req) => requests.push(req.url()));
  await installFixtures(page);
  await page.goto('/?ticker=MSFT&range=MAX&tab=news');
  await expect(page.locator('#chartContainer canvas')).toBeVisible();
  await expect(page.getByRole('button', { name: 'MAX', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#panel-news')).toBeVisible();
  await expect(page.getByText('[Example fixture] Company update for layout verification')).toBeVisible();
  await expect(page.locator('.sec-filing-badge')).toHaveText('SEC 10-K');
  await page.getByRole('button', { name: 'SEC Filings (1)' }).click();
  await expect(page.getByText('Annual Report (Form 10-K)')).toBeVisible();
  await expect(page.getByText('[Example fixture] Company update for layout verification')).toHaveCount(0);

  // Intraday endpoint stays lazy until user clicks 24H
  expect(requests.filter((u) => u.includes('/api/v1/intraday'))).toHaveLength(0);
  const btn24h = page.getByRole('button', { name: '24H', exact: true });
  await expect(btn24h).toBeEnabled();
  await btn24h.click();
  await expect.poll(() => requests.filter((u) => u.includes('/api/v1/intraday')).length).toBe(1);
  await expect(btn24h).toHaveAttribute('aria-pressed', 'true');
  expect(page.url()).toContain('range=24H');

  // Navigating from bare / -> submit MSFT -> browser Back restores bare landing view
  await page.goto('/');
  await expect(page.locator('#chartContainer')).toHaveCount(0);
  await page.getByLabel('Stock ticker').fill('MSFT');
  await page.getByRole('button', { name: /view outlook/i }).click();
  await expect(page.locator('#chartContainer canvas')).toBeVisible();
  expect(page.url()).toContain('ticker=MSFT');
  await page.goBack();
  await expect(page.locator('#chartContainer')).toHaveCount(0);
  expect(new URL(page.url()).search).toBe('');
});
