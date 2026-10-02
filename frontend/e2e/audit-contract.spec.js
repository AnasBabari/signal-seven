import { expect, test } from '@playwright/test';
import { installFixtures, volatilityPayload } from './uiFixtures.js';

async function submit(page) {
  await page.goto('/');
  await page.getByLabel('Stock ticker').fill('MSFT');
  await page.getByRole('button', { name: /view outlook/i }).click();
  await expect(page.locator('#chartContainer canvas')).toBeVisible();
}

test('company search and keyboard selection submit the supported ticker', async ({ page }) => {
  await installFixtures(page);
  await page.route('**/api/v1/search?**', (route) => route.fulfill({ json: { results: [{ ticker: 'MSFT', name: 'Microsoft Corp.' }] } }));
  await page.goto('/');
  const input = page.getByLabel('Stock ticker');
  await input.fill('Microsoft');
  await expect(page.getByRole('option', { name: /Microsoft Corp/ })).toBeVisible();
  await input.press('ArrowDown');
  await input.press('Enter');
  await expect(input).toHaveValue('MSFT');
  await expect(page.locator('#chartContainer canvas')).toBeVisible();
});

test('range controls and date inspection are usable from the keyboard', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await installFixtures(page);
  await submit(page);
  const range = page.getByRole('button', { name: 'MAX', exact: true });
  await range.focus(); await range.press('Enter');
  await expect(range).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText(/Available history/)).toBeVisible();
  const inspector = page.getByRole('slider', { name: 'Inspect chart date' });
  await inspector.focus(); await inspector.press('End');
  await expect(page.locator('output')).toContainText('2026-09-15');
  await page.getByText('Price data table', { exact: true }).click();
  await expect(page.getByRole('table')).toBeVisible();
  await expect(page.getByRole('rowheader', { name: '2026-09-15' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('one never-resolving horizon leaves its loading state while siblings remain visible', async ({ page }) => {
  await installFixtures(page);
  await page.route('**/api/v1/volatility/forecast?**', async (route) => {
    if (new URL(route.request().url()).searchParams.get('horizon') === '10') return new Promise(() => {});
    return route.fallback();
  });
  await submit(page);
  await expect(page.getByText('5 sessions', { exact: true })).toBeVisible();
  await expect(page.getByText('20 sessions', { exact: true })).toBeVisible();
  await expect(page.getByText(/10-session outlook unavailable/)).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText(/10-session outlook loading/)).toHaveCount(0);
});

test('a failed chart chunk exposes retry and the price data table', async ({ page }) => {
  await installFixtures(page);
  await page.route('**/assets/chart-*.js', (route) => route.abort('failed'));
  await page.goto('/');
  await page.getByLabel('Stock ticker').fill('MSFT');
  await page.getByRole('button', { name: /view outlook/i }).click();
  await expect(page.getByRole('button', { name: /Retry chart engine/i })).toBeVisible();
  await page.getByText('Price data table', { exact: true }).click();
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
    if (url.pathname.includes('/volatility/forecast')) {
      const payload = volatilityPayload(Number(url.searchParams.get('horizon')));
      payload.ticker = ticker; payload.current_price = 150;
      return route.fulfill({ json: payload });
    }
    if (url.pathname.endsWith('/forecast')) return route.fulfill({ json: { ticker, current_price: 150, data_as_of: '2026-09-03', future_dates: ['2026-09-04', '2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11', '2026-09-14', '2026-09-15'], lower_prices: Array(7).fill(149), upper_prices: Array(7).fill(153) } });
    if (url.pathname.endsWith('/news')) return route.fulfill({ json: { ticker, status: 'available', items: [] } });
    return route.fulfill({ json: { ticker, entries: [], live_track_record: {}, replay_track_record: {} } });
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

test('desktop and mobile retain readable summary and numerical fallback', async ({ page }) => {
  await installFixtures(page, 10);
  await submit(page);
  await page.screenshot({ path: '../artifacts/audit-ui-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator('.chart-estimate-bar')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: '../artifacts/audit-ui-mobile.png', fullPage: true });
});
