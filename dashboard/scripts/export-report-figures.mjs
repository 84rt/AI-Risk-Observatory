import fs from 'fs';
import path from 'path';
import { spawnSync } from 'child_process';

const BASE_URL = process.env.REPORT_FIGURE_BASE_URL ?? 'http://127.0.0.1:3000';
const OUTPUT_DIR = path.join(process.cwd(), '..', 'report', 'figures');
const REPORT_FIGURES_PATH = path.join(process.cwd(), '..', 'report', 'report-figures.json');
const FIGURE_IDS = ['figure1', 'figure2', 'figure3', 'figure4', 'figure5', 'figure6', 'figure7', 'figure8'];

const CHROME_BIN =
  process.env.CHROME_BIN ??
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const VIEWPORTS = {
  figure1: { width: 1280, height: 760 },
  figure2: { width: 1280, height: 760 },
  figure3: { width: 1280, height: 860 },
  figure4: { width: 1280, height: 820 },
  figure5: { width: 1280, height: 760 },
  figure6: { width: 1280, height: 860 },
  figure7: { width: 1320, height: 1120 },
  figure8: { width: 1280, height: 760 },
};

if (!fs.existsSync(CHROME_BIN)) {
  throw new Error(`Chrome binary not found at ${CHROME_BIN}. Set CHROME_BIN to override.`);
}

if (!fs.existsSync(REPORT_FIGURES_PATH)) {
  throw new Error(`Missing ${REPORT_FIGURES_PATH}. Run npm run build:report-figures first.`);
}

fs.mkdirSync(OUTPUT_DIR, { recursive: true });

for (const figureId of FIGURE_IDS) {
  const viewport = VIEWPORTS[figureId];
  const outputPath = path.join(OUTPUT_DIR, `${figureId}.png`);
  const url = `${BASE_URL}/report-figures/${figureId}`;
  const result = spawnSync(
    CHROME_BIN,
    [
      '--headless=new',
      '--disable-gpu',
      '--no-sandbox',
      '--disable-background-networking',
      '--disable-component-update',
      '--disable-sync',
      '--no-first-run',
      '--hide-scrollbars',
      '--run-all-compositor-stages-before-draw',
      '--force-device-scale-factor=2',
      `--window-size=${viewport.width},${viewport.height}`,
      '--virtual-time-budget=10000',
      `--screenshot=${outputPath}`,
      url,
    ],
    {
      encoding: 'utf8',
    }
  );

  if (result.status !== 0) {
    throw new Error([
      `Failed to export ${figureId}.`,
      result.stdout?.trim(),
      result.stderr?.trim(),
    ].filter(Boolean).join('\n'));
  }

  const stats = fs.statSync(outputPath);
  console.log(`Wrote ${outputPath} (${stats.size} bytes)`);
}
