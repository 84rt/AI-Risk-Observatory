import fs from 'fs';
import path from 'path';

import {
  ANNOTATIONS_PATH,
  PRECOMPUTED_DASHBOARD_DATA_PATH,
  buildGoldenSetDashboardDataFromRaw,
} from '../src/lib/golden-set.ts';

const outputDir = path.dirname(PRECOMPUTED_DASHBOARD_DATA_PATH);

if (!fs.existsSync(ANNOTATIONS_PATH)) {
  console.error(`Missing raw annotations file: ${ANNOTATIONS_PATH}`);
  process.exit(1);
}

fs.mkdirSync(outputDir, { recursive: true });

const data = buildGoldenSetDashboardDataFromRaw();

const validateFilterIndexYears = (rows, label) => {
  const invalidRow = rows.find(row => !Number.isInteger(row[0]));
  if (invalidRow) {
    throw new Error(`Invalid ${label} filterIndex row year: ${JSON.stringify(invalidRow)}`);
  }
};

if (!data.filterIndex?.perReport || !data.filterIndex?.perChunk) {
  throw new Error('Dashboard data is missing filterIndex rows.');
}
validateFilterIndexYears(data.filterIndex.perReport, 'perReport');
validateFilterIndexYears(data.filterIndex.perChunk, 'perChunk');

fs.writeFileSync(PRECOMPUTED_DASHBOARD_DATA_PATH, `${JSON.stringify(data)}\n`, 'utf8');

const stats = fs.statSync(PRECOMPUTED_DASHBOARD_DATA_PATH);
console.log(`Wrote ${PRECOMPUTED_DASHBOARD_DATA_PATH} (${stats.size} bytes)`);
