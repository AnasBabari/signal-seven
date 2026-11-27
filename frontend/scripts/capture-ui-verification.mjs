import { chromium, devices } from '@playwright/test';
import { preview } from 'vite';

import { installFixtures } from '../e2e/uiFixtures.js';

async function submit(page) {
  await page.goto('/');
  await page.getByLabel('Stock ticker').fill('MSFT');
  await page.getByRole('button', { name: /view outlook/i }).click();
  await page.getByText(/^80% pointwise conformal range:/i).waitFor({ state: 'visible' });
  await page.getByRole('heading', { name: 'Volatility Outlook' }).waitFor();
  await page.getByRole('heading', { name: 'Latest news & filings' }).waitFor();
  await page.locator('#chartContainer canvas').waitFor({ state: 'visible' });
  // Allow the chart's documented 250ms transition to complete before capture.
  await page.waitForTimeout(500);
}

function installCaptureFixtures(page, failedHorizon = null) {
  return installFixtures(page, failedHorizon, {
    model: { name: 'ridge' },
    interval_calibration: { method: 'volatility_scaled_online_split_conformal', m5_gate_verdict: 'PASS' },
  }, 'har_rv');
}

const browser = await chromium.launch({ headless: true });
let server;
try {
  server = await preview({ preview: { host: '127.0.0.1', port: 0 } });
  const baseURL = `http://127.0.0.1:${server.httpServer.address().port}`;
  const desktop = await browser.newPage({ viewport: { width: 1440, height: 1000 }, baseURL });
  await installCaptureFixtures(desktop);
  await submit(desktop);
  await desktop.screenshot({ path: '../docs/verification/ui-success-desktop.png', fullPage: true });
  await desktop.getByRole('button', { name: /expand chart/i }).click();
  await desktop.getByRole('dialog', { name: /expanded price chart/i }).waitFor();
  await desktop.waitForTimeout(500);
  await desktop.screenshot({ path: '../docs/verification/ui-expanded-desktop.png' });
  await desktop.keyboard.press('Escape');
  await desktop.close();

  const mobile = await browser.newPage({ ...devices['iPhone 13'], baseURL });
  await installCaptureFixtures(mobile);
  await submit(mobile);
  await mobile.screenshot({ path: '../docs/verification/ui-success-mobile.png', fullPage: true });
  await mobile.close();

  const partial = await browser.newPage({ viewport: { width: 1440, height: 1000 }, baseURL });
  await installCaptureFixtures(partial, 10);
  await submit(partial);
  await partial.getByText(/10-session outlook unavailable/i).waitFor({ state: 'visible', timeout: 15_000 });
  await partial.screenshot({ path: '../docs/verification/ui-partial-volatility-desktop.png', fullPage: true });
  await partial.close();

  const partialMobile = await browser.newPage({ ...devices['iPhone 13'], baseURL });
  await installCaptureFixtures(partialMobile, 10);
  await submit(partialMobile);
  await partialMobile.getByText(/10-session outlook unavailable/i).waitFor();
  await partialMobile.screenshot({ path: '../docs/verification/ui-partial-volatility-mobile.png', fullPage: true });
  await partialMobile.close();
} finally {
  await browser.close();
  await server?.close();
}
