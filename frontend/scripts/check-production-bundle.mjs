import { readdir, readFile, stat } from 'node:fs/promises';
import { join } from 'node:path';

const MAX_CHART_ENGINE_BYTES = 175_000;
const MAX_CSS_BYTES = 36_000;

const assets = join(process.cwd(), 'dist', 'assets');
const files = await readdir(assets);
const forbiddenNames = files.filter((name) =>
  /trainingWorker|browserTrainingClient|tfjs|jszip/i.test(name),
);

const forbiddenContents = [];
for (const name of files.filter((entry) => entry.endsWith('.js'))) {
  const content = await readFile(join(assets, name), 'utf8');
  if (
    content.includes('TensorFlow.js')
    || content.includes('indexeddb://stocklstm')
    || content.includes('tfjs-return-lstm')
  ) {
    forbiddenContents.push(name);
  }
}

if (forbiddenNames.length || forbiddenContents.length) {
  throw new Error(
    `production bundle contains forbidden legacy code; names=${forbiddenNames.join(',') || 'none'} contents=${forbiddenContents.join(',') || 'none'}`,
  );
}

const chartChunks = files.filter((name) => /^chart-engine.*\.js$/i.test(name));
if (chartChunks.length !== 1) {
  throw new Error(`expected exactly 1 lazy chart-engine chunk in dist/assets, found: ${chartChunks.join(', ') || 'none'}`);
}
const chartStat = await stat(join(assets, chartChunks[0]));
if (chartStat.size > MAX_CHART_ENGINE_BYTES) {
  throw new Error(
    `${chartChunks[0]} size ${chartStat.size} bytes exceeds tree-shaken ceiling of ${MAX_CHART_ENGINE_BYTES} bytes`,
  );
}

const cssFiles = files.filter((name) => name.endsWith('.css'));
if (cssFiles.length === 0) {
  throw new Error('expected at least 1 CSS asset in dist/assets');
}
let totalCssBytes = 0;
for (const name of cssFiles) {
  const cssContent = await readFile(join(assets, name), 'utf8');
  if (cssContent.includes('[data-theme=light]') || cssContent.includes('[data-theme="light"]')) {
    throw new Error(`${name} still contains dead [data-theme="light"] rules`);
  }
  totalCssBytes += (await stat(join(assets, name))).size;
}
if (totalCssBytes > MAX_CSS_BYTES) {
  throw new Error(
    `CSS bundle size ${totalCssBytes} bytes exceeds purged ceiling of ${MAX_CSS_BYTES} bytes`,
  );
}

console.log(
  `Production bundle OK (${files.length} assets): ${chartChunks[0]}=${chartStat.size} B (cap ${MAX_CHART_ENGINE_BYTES} B), css=${totalCssBytes} B (cap ${MAX_CSS_BYTES} B), TFJS/jszip/light-theme-free.`,
);
